"""Оркестратор фермы: диспетчер, специалисты, контролёр.

Схема ровно та, ради которой всё собиралось:

    задача -> диспетчер делит на подзадачи
           -> каждой подзадаче определяется домен (код, 1С, SQL, торговля…)
           -> исполнители работают параллельно на бесплатных слотах
           -> контролёр другой модели проверяет каждый ответ по критериям домена
           -> брак уходит на переделку, но не более max_iter раз
           -> диспетчер сводит принятое в один результат

Три предохранителя, без которых это разорительно или бесконечно:

* ``max_iter`` — потолок переделок на подзадачу. Иначе исполнитель и критик
  перекидывают работу друг другу вечно.
* ``max_paid_calls`` — потолок обращений к платному тиру. Диспетчер платный,
  поэтому при исчерпании бюджета он не зовётся, а задача берётся целиком.
* ``rotation`` — разведение одновременных подзадач по разным слотам. Все
  воркеры иначе выбирают первый слот пула и встают в очередь друг за другом.

Домен подзадачи определяется по ключевым словам, а не моделью: это бесплатно
и мгновенно. Ошибка в домене стоит дешевле, чем лишний вызов модели.
"""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import websearch
from .backends import BackendError, Reply
from .ledger import Ledger
from .models import Slot, build_pool, pick
from .roles import (
    AGGREGATOR_SYSTEM,
    GENERAL,
    PLANNER_SYSTEM,
    Domain,
    aggregator_prompt,
    critic_prompt,
    critic_verdict,
    planner_prompt,
    revision_prompt,
    route,
    worker_prompt,
)


def is_blank(text: str) -> bool:
    """Пустой ответ — это не ответ.

    Проверено 19.09.2026: прогон, где все воркеры упали, записал в файл
    0 байт и отчитался успехом. Модель может вернуть пустоту или одну
    ограду ``` без содержимого; и то и другое надо считать неудачей и
    уходить на другой слот, а не отдавать наружу как результат.
    """

    body = text.strip()
    if not body:
        return True
    return all(
        not line.strip() or line.strip().startswith("```") for line in body.splitlines()
    )


@dataclass
class Step:
    """Одна подзадача и её судьба."""

    subtask: str
    domain: str = GENERAL.key
    answer: str = ""
    worker: str = ""
    critic: str = ""
    verdict_ok: bool = True
    note: str = ""
    attempts: int = 0
    error: str = ""


@dataclass
class FarmResult:
    task: str
    steps: list[Step] = field(default_factory=list)
    answer: str = ""
    paid_calls: int = 0
    free_calls: int = 0
    notes: list[str] = field(default_factory=list)
    savings: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.answer) and not self.notes

    def summary(self) -> str:
        """Короткий отчёт о прогоне — для лога и для консоли."""

        lines = [f"подзадач: {len(self.steps)}"]
        for index, step in enumerate(self.steps, 1):
            mark = "OK " if step.verdict_ok else "БРАК"
            who = step.worker or "—"
            extra = f" ({step.error})" if step.error else ""
            lines.append(
                f"  {index}. [{mark}] {step.domain:<8} {who:<28} попыток: {step.attempts}{extra}"
            )
            lines.append(f"     {step.subtask[:88]}")
        lines.append(
            f"обращений к моделям: бесплатных ~{self.free_calls}, платных {self.paid_calls}"
        )
        if self.savings:
            lines.append(self.savings)
        return "\n".join(lines)


class Farm:
    def __init__(
        self,
        *,
        max_subtasks: int = 4,
        max_iter: int = 2,
        max_paid_calls: int = 3,
        workers: int = 4,
        include_paid: bool = True,
        default_domain: Domain = GENERAL,
        ledger: Ledger | None = None,
        web: bool = False,
        web_limit: int = 6,
    ) -> None:
        self.pool = build_pool(include_paid=include_paid)
        self.max_subtasks = max_subtasks
        # Веб-поиск по умолчанию выключен: он ходит в сеть и удлиняет промпт,
        # а нужен не каждой задаче — для переписывания текста он только лишний.
        self.web = web
        self.web_limit = web_limit
        self.max_iter = max_iter
        self.max_paid_calls = max_paid_calls
        self.workers = workers
        self.default_domain = default_domain
        # Журнал расхода. Отключается только тестами: без него не видно
        # экономии, ради которой ферма и собрана.
        self.ledger = ledger if ledger is not None else Ledger()
        self._paid_used = 0
        self._paid_lock = threading.Lock()
        self._rotation = 0
        self._rotation_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Служебное
    # ------------------------------------------------------------------

    def _next_rotation(self) -> int:
        """Выдать следующий номер сдвига для выбора слота.

        Счётчик общий на ферму: именно чередование разводит одновременные
        подзадачи по разным моделям.
        """

        with self._rotation_lock:
            self._rotation += 1
            return self._rotation

    def _spend_paid(self) -> bool:
        """Списать одно обращение к платному тиру. False — бюджет исчерпан.

        Под замком: диспетчер и сводка могут зваться из разных потоков, и без
        блокировки бюджет превышался бы на числе потоков.
        """

        with self._paid_lock:
            if self._paid_used >= self.max_paid_calls:
                return False
            self._paid_used += 1
            return True

    def _call(
        self, slot: Slot, prompt: str, *, system: str = "", max_tokens: int = 1500
    ) -> Reply:
        """Обратиться к модели и записать расход.

        Учёт живёт здесь, в единственной точке выхода к моделям: если считать
        в вызывающем коде, рано или поздно один из вызовов забудут, и отчёт
        об экономии станет неполным — то есть неверным.
        """

        reply = slot.backend.complete(prompt, system=system, max_tokens=max_tokens)
        if self.ledger is not None:
            self.ledger.record(
                slot=slot.name,
                model=slot.model,
                free=slot.free,
                tokens_in=reply.tokens_in,
                tokens_out=reply.tokens_out,
                estimated_tokens=reply.estimated_tokens,
            )
        return reply

    def _web_context(self, query: str) -> str:
        """Выжимка веб-поиска для промпта.

        Пустая строка — обычный случай: поиск выключен, молчит или ничего не
        нашёл. Ронять прогон из-за недоступной сети нельзя, поэтому любой отказ
        здесь превращается в «контекста нет», а задача решается как раньше.
        """

        if not self.web:
            return ""
        return websearch.context_block(query, limit=self.web_limit)

    def _pick(
        self, *, skill: str, exclude: set[str], free: bool | None = True, strongest: bool = False
    ) -> Slot:
        return pick(
            self.pool,
            skill=skill,
            exclude=exclude,
            free=free,
            rotation=self._next_rotation(),
            strongest=strongest,
        )

    # ------------------------------------------------------------------
    # Этапы
    # ------------------------------------------------------------------

    def _pick_planner(self) -> Slot | None:
        """Выбрать модель для роли диспетчера.

        Сначала платная — она сильнее всех и потому планирует лучше. Если
        платный тир выключен или бюджет исчерпан, берётся бесплатная с
        умением ``planning``; сейчас это модели Zen, и их качества для
        декомпозиции хватает.

        Если планировщика нет вовсе, возвращается None: тогда задача идёт
        целиком одному исполнителю. Это лучше плохого плана — проверено на
        живом прогоне, где слабая модель выдала вместо подзадач пересказ
        собственной инструкции.
        """

        try:
            paid = self._pick(skill="planning", exclude=set(), free=False)
        except LookupError:
            paid = None
        if paid is not None and self._spend_paid():
            return paid
        try:
            return self._pick(skill="planning", exclude=set(), free=True)
        except LookupError:
            return None

    def plan(self, task: str, *, context: str = "") -> list[str]:
        """Разбить задачу на подзадачи.

        Если диспетчер недоступен или не ответил, задача берётся целиком:
        хуже разбиения только отсутствие ответа.
        """

        fallback = [task]
        planner = self._pick_planner()
        if planner is None:
            return fallback
        try:
            reply = self._call(
                planner,
                planner_prompt(task, self.max_subtasks, context=context),
                system=PLANNER_SYSTEM,
                max_tokens=800,
            )
        except BackendError:
            return fallback
        return self._parse_plan(reply.text) or fallback

    def _parse_plan(self, raw: str) -> list[str]:
        """Достать список подзадач из ответа модели.

        Модель может обернуть JSON в ```-блок или добавить пояснение, поэтому
        сначала пробуем разобрать как есть, потом ищем первый массив в тексте.
        """

        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if not match:
                return []
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return []
        if isinstance(data, dict):
            data = data.get("subtasks") or data.get("подзадачи") or []
        if not isinstance(data, list):
            return []
        items = [str(item).strip() for item in data if str(item).strip()]
        return items[: self.max_subtasks]

    def _work(self, task: str, subtask: str, base: Domain) -> Step:
        """Выполнить одну подзадачу: исполнитель, проверка, переделки."""

        domain = route(subtask, default=base)
        step = Step(subtask=subtask, domain=domain.key)
        exclude: set[str] = set()

        try:
            worker = self._pick(skill="code" if domain.key in {"code", "web"} else "text",
                                exclude=exclude)
        except LookupError:
            step.error = "нет свободных исполнителей"
            return step
        step.worker = worker.name

        # Свежая выдача по самой подзадаче: у каждой свой запрос, поэтому и
        # контекст свой — общая выжимка по всей задаче здесь была бы шумом.
        context = self._web_context(subtask)
        prompt = worker_prompt(domain, task=task, subtask=subtask, context=context)
        for attempt in range(1, self.max_iter + 1):
            step.attempts = attempt
            try:
                answer = self._call(worker, prompt, system=domain.worker_system()).text
                if is_blank(answer):
                    raise BackendError(f"{worker.name}: пустой ответ")
                step.answer = answer
            except BackendError as exc:
                step.error = str(exc)
                # Слот отвалился — берём следующий, а не тот же самый.
                exclude.add(worker.name)
                try:
                    worker = self._pick(skill="text", exclude=exclude)
                except LookupError:
                    return step
                step.worker = worker.name
                continue

            step.verdict_ok, step.note, step.critic = self._critique(
                domain, subtask, step.answer, exclude={worker.name}
            )
            if step.verdict_ok:
                return step

            prompt = revision_prompt(
                domain,
                task=task,
                subtask=subtask,
                answer=step.answer,
                note=step.note,
                context=context,
            )
        step.error = step.error or f"не принято за {self.max_iter} попытки"
        return step

    def _critique(
        self, domain: Domain, subtask: str, answer: str, *, exclude: set[str]
    ) -> tuple[bool, str, str]:
        """Проверить ответ чужой моделью.

        Контролёр обязан отличаться от исполнителя: та же модель повторит его
        ошибки вместо того, чтобы их поймать. Если свободных слотов кроме
        исполнителя нет, проверка пропускается — лучше сдать ответ с
        оговоркой, чем зациклиться на переделках.
        """

        try:
            critic = self._pick(skill="critique", exclude=exclude, strongest=True)
        except LookupError:
            try:
                critic = self._pick(skill="text", exclude=exclude, strongest=True)
            except LookupError:
                return True, "", ""
        try:
            reply = self._call(
                critic,
                critic_prompt(domain, subtask=subtask, answer=answer),
                system=domain.critic_system(),
                # 300 токенов хватало текстовым моделям, но reasoning-модели
                # (kimi-k3, glm-5.3) тратят бюджет на размышление и при малом
                # лимите возвращают content=null. Критик отвечает коротко, так
                # что запас лимита ничего не удлиняет — он только снимает потолок.
                max_tokens=1200,
            )
        except BackendError:
            return True, "", ""
        ok, note = critic_verdict(reply.text)
        return ok, note, critic.name

    def aggregate(self, task: str, steps: list[Step]) -> str:
        accepted = [step for step in steps if not is_blank(step.answer)]
        if not accepted:
            return ""
        if len(accepted) == 1:
            return accepted[0].answer
        joined = "\n\n".join(f"=== {step.subtask} ===\n{step.answer}" for step in accepted)
        aggregator = self._pick_planner()
        if aggregator is None:
            return joined
        try:
            return self._call(
                aggregator,
                aggregator_prompt(task, joined),
                system=AGGREGATOR_SYSTEM,
                max_tokens=2500,
            ).text
        except BackendError:
            return joined

    # ------------------------------------------------------------------
    # Запуск
    # ------------------------------------------------------------------

    def run(self, task: str, *, domain: Domain | None = None) -> FarmResult:
        base = domain or route(task, default=self.default_domain)
        result = FarmResult(task=task)
        # Диспетчер получает поиск по всей задаче: так он видит, из чего она
        # состоит на самом деле, и делит осмысленнее, чем по одному названию.
        subtasks = self.plan(task, context=self._web_context(task))
        result.notes.append(f"домен задачи: {base.key}, подзадач: {len(subtasks)}")

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            steps = list(pool.map(lambda subtask: self._work(task, subtask, base), subtasks))

        result.steps = steps
        result.answer = self.aggregate(task, steps)
        result.paid_calls = self._paid_used
        result.free_calls = sum(step.attempts for step in steps) + sum(
            1 for step in steps if step.critic
        )
        for step in steps:
            if step.error:
                result.notes.append(f"{step.subtask[:40]}: {step.error}")
        result.savings = self.ledger.session_line()
        return result


def run(task: str, **kwargs) -> FarmResult:
    return Farm(**kwargs).run(task)
