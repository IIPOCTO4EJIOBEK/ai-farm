"""Реестр моделей фермы: кто бесплатный, кто платный и на что годится.

Цены здесь не в деньгах, а в признаке ``free``: ферма обязана сначала
израсходовать бесплатные слоты и только потом звать платный тир. Слот — это
пара «аккаунт x модель»: у аккаунтов Zen свои лимиты, поэтому 7 моделей на
двух аккаунтах дают 14 независимых слотов, а не 7.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .backends import (
    Backend,
    FccBackend,
    NvidiaBackend,
    OllamaBackend,
    ZenBackend,
    nvidia_key,
)

# Бесплатные модели Zen. Перемерено 27.09.2026 на живом куске стенограммы
# (11 746 знаков): big-pickle 12–30 с, ling-3.0-flash-fin 13 с,
# muse-spark-1.3 27 с, nemotron-3-ultra 17 с.
#
# Убраны две мёртвые — mimo-v2.5-free и muse-spark-1.2-contributor-free:
# обе на обоих аккаунтах отвечают UnknownError «Unexpected server error»
# за 2.3 с. Пока они лежали в пуле, каждый кусок digest'а платил за них
# лишним отказом и ходом на слот подмены.
ZEN_FREE_MODELS = (
    "big-pickle",
    "ling-3.0-flash-fin-free",
    "muse-spark-1.3-contributor-free",
    "nemotron-3-ultra-free",
    "nemotron-3.5-lightning-free",
)

ZEN_ACCOUNTS = ("A", "B")

# Слоты, которые не поднимать, хотя модель жива. Ключ — пара
# «аккаунт x модель», потому что ломается именно слот, а не модель:
# nemotron-3.5-lightning-free на аккаунте A зависает (261 с, затем потолок
# 300 с), а на B отвечает за 18 с. Выкинув модель целиком, я потерял бы
# рабочий слот B. Проверено 27.09.2026. Отсюда же брались 14 минут, за
# которые digest проходил восемь кусков: он ждал потолок 420 с на этом слоте.
ZEN_SKIP = frozenset({("A", "nemotron-3.5-lightning-free")})

# Локальные модели ноута. Замеры: qwen2.5:3b ~14.8 ток/с,
# qwen2.5-coder:7b ~7.0 ток/с, qwen3:4b тратит токены на размышления.
LOCAL_MODELS = ("qwen2.5:3b", "qwen2.5-coder:7b")

# Модели NVIDIA NIM. Фри-тир там не про кредиты, а про ~40 запросов в
# минуту на ключ, поэтому это не «ещё один бесплатный тир», а способ снять
# нагрузку с Zen: когда аккаунт Zen упирается в лимит, задача уходит сюда.
# Слоты появляются в пуле только при наличии ключа — без него каждый вызов
# был бы честным отказом и только засорял бы учёт.
NVIDIA_MODELS = (
    "deepseek-ai/deepseek-v4.1-flash",
    "moonshotai/kimi-k3",
    "z-ai/glm-5.3",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "mistralai/mistral-large-2-instruct",
    "openai/gpt-oss-20b",
)


@dataclass(frozen=True)
class Slot:
    """Одно место, куда можно отдать работу."""

    name: str
    backend: Backend
    free: bool
    skills: frozenset[str] = field(default_factory=frozenset)
    # ``tier`` и ``quality`` — разные оси, и склеивать их нельзя.
    #
    # tier — очерёдность: чем ниже, тем раньше берут. Про цену и про то, куда
    # девать работу в первую очередь.
    # quality — сила модели: чем выше, тем лучше она соображает. Про то, кому
    # доверить проверку и декомпозицию.
    #
    # Пока это было одно поле, ферма отправляла работу сначала в локальную
    # 3B-модель — она «дешевле» — а она и слабее бесплатных облачных, и в
    # разы медленнее: 7 ток/с против сетевого вызова. Роль контролёра при
    # этом доставалась самой слабой модели в пуле.
    tier: int = 0
    quality: int = 1

    # Имя модели у провайдера — по нему учёт расхода находит ставку.
    model: str = ""

    def __str__(self) -> str:
        mark = "free" if self.free else "PAID"
        return f"{self.name} [{mark}]"


def build_pool(*, include_paid: bool = True) -> list[Slot]:
    """Собрать все доступные слоты."""

    pool: list[Slot] = []

    for model in LOCAL_MODELS:
        # Умения planning у локальных моделей нет намеренно. Проверено на
        # живом прогоне: qwen2.5:3b в роли диспетчера вернула вместо подзадач
        # пересказ собственной инструкции — «сформулируй повелительным
        # наклонением подзадачу». План получился мусорным, и вся работа
        # ушла в брак. Разделить задачу хуже сильной модели, но пересказать
        # инструкцию вместо плана — это не разделение вовсе.
        skills = {"code", "text"} if "coder" in model else {"text", "short"}
        pool.append(
            Slot(
                name=f"ollama/{model}",
                backend=OllamaBackend(model),
                free=True,
                skills=frozenset(skills),
                # Локальные модели — подстраховка, а не первый выбор: они
                # слабее бесплатных облачных и медленнее их в разы (инференс
                # идёт на CPU). К ним обращаются, когда Zen недоступен или
                # упёрся в лимит, либо когда нужна работа без сети.
                tier=1,
                quality=1,
                model=model,
            )
        )

    for account in ZEN_ACCOUNTS:
        for model in ZEN_FREE_MODELS:
            if (account, model) in ZEN_SKIP:
                continue
            pool.append(
                Slot(
                    name=f"zen{account}/{model}",
                    backend=ZenBackend(model, account),
                    free=True,
                    skills=frozenset({"text", "code", "critique", "planning"}),
                    # Первый выбор: бесплатно и качественнее локальных.
                    # Аккаунтов два, поэтому 7 моделей дают 14 независимых
                    # слотов с раздельными лимитами.
                    tier=0,
                    quality=2,
                    model=model,
                )
            )

    if nvidia_key():
        for model in NVIDIA_MODELS:
            pool.append(
                Slot(
                    name=f"nvidia/{model}",
                    backend=NvidiaBackend(model),
                    free=True,
                    skills=frozenset({"text", "code", "critique", "planning"}),
                    # Очерёдность та же, что у Zen: бесплатно. Отдельный тир
                    # нужен не для порядка, а для учёта — видно, сколько
                    # задач ушло в NVIDIA, а сколько в Zen.
                    tier=0,
                    # Сильнее среднего Zen-слота: это крупные модели
                    # (253B, kimi-k3, glm-5.3), а не облегчённые версии.
                    quality=3,
                    model=model,
                )
            )

    if include_paid:
        # Через шлюз FCC: имена моделей с подстрокой sonnet/opus роутер FCC
        # разводит по тирам, а списываются деньги за модель из ~/.fcc/.env
        # (MODEL_SONNET / MODEL_OPUS). Ставка в учёте берётся по этим именам.
        # sonnet — дешевле, поэтому берётся первым среди платных; opus — самый
        # сильный в пуле и потому последний по очерёдности, но первый по
        # качеству. Диспетчером становится sonnet: для декомпозиции его
        # хватает, а платить за opus на каждой задаче незачем.
        pool.append(
            Slot(
                name="fcc/claude-sonnet-4-5",
                backend=FccBackend("claude-sonnet-4-5"),
                free=False,
                skills=frozenset({"text", "code", "reasoning", "planning"}),
                tier=2,
                quality=3,
                model="deepseek/deepseek-v4-flash",
            )
        )
        pool.append(
            Slot(
                name="fcc/claude-opus-4-1",
                backend=FccBackend("claude-opus-4-1"),
                free=False,
                skills=frozenset({"reasoning", "planning", "critique"}),
                tier=3,
                quality=4,
                model="deepseek/deepseek-v4-pro",
            )
        )

    return pool


def free_slots(pool: list[Slot]) -> list[Slot]:
    return [slot for slot in pool if slot.free]


def pick(
    pool: list[Slot],
    *,
    skill: str | None = None,
    exclude: set[str] | None = None,
    free: bool | None = True,
    rotation: int = 0,
    strongest: bool = False,
) -> Slot:
    """Выбрать слот под задачу.

    Порядок предпочтений: сначала дешёвый тир, потом более высокий. Так
    бесплатные слоты расходуются раньше платных, а платный зовут только когда
    бесплатных подходящих не осталось.

    ``free`` отбирает по цене явно: ``True`` — только бесплатные, ``False`` —
    только платные, ``None`` — любые. Именно ``False`` нужен диспетчеру:
    «неважно какой цены» и «только платный» — разные вещи, и без этого
    различия планировщиком становилась локальная 3B-модель.

    ``rotation`` сдвигает выбор внутри самого дешёвого тира. Без этого все
    параллельные подзадачи выбирают один и тот же первый слот: четыре
    воркера выстраиваются в очередь к одной модели и параллелизм
    превращается в ожидание. Сдвиг сохраняет правило «сначала дешёвое», но
    разводит одновременные задачи по разным моделям.

    ``strongest`` переключает сортировку с очерёдности на качество: берётся
    самая сильная модель, а не самая дешёвая. Это для роли контролёра —
    проверено на живом прогоне, где контролёр на 3B-модели забраковал верный
    код, написанный 7B: судья слабее подсудимого не судит, а придирается.
    """

    exclude = exclude or set()
    candidates = [
        slot
        for slot in pool
        if slot.name not in exclude and (free is None or slot.free == free)
    ]
    # Слоты, поймавшие лимит, пропускаем сразу: они всё равно откажут, но
    # откажут не мгновенно, а через таймаут. Если же в строю не осталось
    # никого, берём как есть — лучше попробовать и получить отказ, чем
    # объявить, что слотов нет.
    ready = [slot for slot in candidates if slot.backend.available()]
    if ready:
        candidates = ready
    if skill:
        skilled = [slot for slot in candidates if skill in slot.skills]
        if skilled:
            candidates = skilled
    if not candidates:
        raise LookupError("нет подходящего слота: все исключены или пул пуст")
    # Сортировка по модели, а уже потом по имени слота: имена начинаются с
    # аккаунта («zenA/…», «zenB/…»), и при сортировке по имени подряд шли бы
    # семь моделей одного аккаунта. Чередование тогда разводило бы задачи по
    # моделям, но не по аккаунтам, — а лимиты у аккаунтов раздельные, и все
    # воркеры били бы в одну квоту. Порядок «модель, затем аккаунт» даёт на
    # соседних номерах разные аккаунты.
    if strongest:
        candidates.sort(key=lambda slot: (-slot.quality, slot.model, slot.name))
        edge_value = candidates[0].quality
        edge = lambda slot: slot.quality == edge_value  # noqa: E731
    else:
        candidates.sort(key=lambda slot: (slot.tier, slot.model, slot.name))
        edge_value = candidates[0].tier
        edge = lambda slot: slot.tier == edge_value  # noqa: E731

    # Чередование внутри верхней группы: без него все одновременные подзадачи
    # выбирают один и тот же первый слот и встают в очередь друг за другом.
    if rotation:
        group = [slot for slot in candidates if edge(slot)]
        if len(group) > 1:
            chosen = group[rotation % len(group)]
            candidates = [chosen] + [slot for slot in candidates if slot is not chosen]
    return candidates[0]
