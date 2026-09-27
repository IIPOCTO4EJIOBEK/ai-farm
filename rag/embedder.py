"""Векторизация текста моделью e5 через ONNX Runtime.

Выбор модели: `intfloat/multilingual-e5-large`, 1024 измерения.

- Машина без CUDA (встроенная Radeon), поэтому нужен рантайм, хорошо
  работающий на CPU. ONNX Runtime под fastembed быстрее torch на этой задаче
  и не тянет за собой гигабайты зависимостей.
- Модель многоязычная и заметно сильнее на русском, чем английские
  `nomic-embed-text` или `all-MiniLM`, на которых всё обычно и строится.
- `bge-m3` (568M параметров) дал бы лучшее качество, но на ноутбучном CPU
  считается в разы дольше — для локального индекса это плохой размен.

Важная деталь семейства e5: модель обучена на префиксах `query: ` и
`passage: `. Без них качество поиска падает заметно, причём молча — ошибка не
проявляется, просто результаты становятся хуже.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from pathlib import Path

DEFAULT_MODEL = "intfloat/multilingual-e5-large"
EMBEDDING_DIM = 1024

# Репозиторий с ONNX-версией модели. fastembed берёт её же, но раскладывает
# через кэш HuggingFace — см. ensure_model_dir, почему нам это не подходит.
HF_REPO = "qdrant/multilingual-e5-large-onnx"

DEFAULT_MODEL_DIR = Path(
    os.environ.get(
        "RAG_MODEL_DIR",
        Path.home() / ".cache" / "ai-workspace" / "models" / "multilingual-e5-large",
    )
)

# Файлы, без которых модель не загрузится.
MODEL_FILES = (
    "model.onnx",
    "model.onnx_data",
    "tokenizer.json",
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def ensure_model_dir(target: Path = DEFAULT_MODEL_DIR, repo: str = HF_REPO) -> Path:
    """Скачать модель в обычный каталог и вернуть путь к нему.

    Зачем не отдать это fastembed. Кэш HuggingFace хранит файлы в `blobs/<хеш>`
    и подставляет на них символьные ссылки, а `model.onnx` и `model.onnx_data`
    из-за разных хешей оказываются в разных каталогах. onnxruntime (начиная с
    1.19) после разыменования ссылки проверяет, что внешние данные лежат рядом
    с самой моделью, видит выход за пределы каталога и отказывается грузить:

        External data path escapes model directory

    Копирование в обычный каталог снимает вопрос: оба файла лежат рядом.

    Заодно каталог постоянный. По умолчанию fastembed складывает модель в
    /tmp, а он чистится при перезагрузке — и 2.2 ГБ пришлось бы качать снова.
    """

    if all((target / name).is_file() for name in MODEL_FILES):
        return target

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # noqa: TRY003
        raise RuntimeError(
            "не установлен huggingface_hub — он нужен для скачивания модели"
        ) from exc

    target.mkdir(parents=True, exist_ok=True)
    print(f"качаю модель {repo} в {target} (около 2.2 ГБ, один раз)...")
    snapshot_download(
        repo_id=repo,
        local_dir=str(target),
        allow_patterns=list(MODEL_FILES),
    )

    missing = [name for name in MODEL_FILES if not (target / name).is_file()]
    if missing:
        raise RuntimeError(
            f"в {target} не хватает файлов: {', '.join(missing)}. "
            "Удалите каталог и повторите — возможно, скачивание оборвалось."
        )
    return target

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# Сколько текстов подавать модели за раз. Больше — быстрее, но заметнее
# расход памяти; на 14 ГБ и 800-символьных фрагментах 32 проходит спокойно.
DEFAULT_BATCH = 32


class Embedder:
    """Обёртка над fastembed с правильными префиксами и нормализацией."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        cache_dir: Path | str | None = None,
        threads: int | None = None,
        batch_size: int = DEFAULT_BATCH,
    ) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # noqa: TRY003
            raise RuntimeError(
                "не установлен fastembed. Выполните: .venv/bin/pip install fastembed"
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        kwargs: dict = {}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        if threads is not None:
            kwargs["threads"] = threads

        # Для модели по умолчанию указываем заранее скачанный каталог: см.
        # ensure_model_dir — через кэш HuggingFace onnxruntime её не примет.
        if model_name == DEFAULT_MODEL:
            kwargs["specific_model_path"] = str(ensure_model_dir())

        # fastembed предупреждает, что модель перешла с CLS-пулинга на mean
        # pooling. Для нас это неважно: индексируем и ищем одной и той же
        # моделью, поэтому вектора заведомо согласованы между собой. Смешать
        # старые и новые можно было бы только при смене версии fastembed на
        # уже проиндексированной базе — тогда нужен --force.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*mean pooling.*")
            self._model = TextEmbedding(model_name, **kwargs)

    # -- публичный интерфейс -------------------------------------------------

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        """Вектора для того, что кладём в базу."""

        return self._embed(texts, PASSAGE_PREFIX)

    def embed_query(self, text: str) -> list[float]:
        """Вектор для поискового запроса. Префикс другой — это не опечатка."""

        result = self._embed([text], QUERY_PREFIX)
        return result[0]

    # -- внутреннее ----------------------------------------------------------

    def _embed(self, texts: Sequence[str], prefix: str) -> list[list[float]]:
        if not texts:
            return []
        prepared = [prefix + self._prepare(t) for t in texts]
        vectors = [self._normalise(v) for v in self._model.embed(prepared, batch_size=self.batch_size)]
        self._check_dim(vectors)
        return vectors

    @staticmethod
    def _prepare(text: str) -> str:
        """Привести текст к виду, пригодному для модели.

        Переводы строк и длинные цепочки пробелов схлопываются: для векторного
        поиска они смысла не несут, а вход занимают.
        """

        return " ".join(text.split())

    @staticmethod
    def _normalise(vector) -> list[float]:
        """Привести вектор к единичной длине.

        В схеме используется косинусное расстояние, а оно нормирует само, так
        что на результат это не влияет. Но длина 1 делает вектора сравнимыми
        скалярным произведением, если позже понадобится другой оператор.
        """

        import math

        values = [float(x) for x in vector]
        norm = math.sqrt(sum(v * v for v in values))
        if norm == 0.0:
            return values
        return [v / norm for v in values]

    @staticmethod
    def _check_dim(vectors: list[list[float]]) -> None:
        """Размерность должна совпадать со схемой, иначе вставка упадёт.

        Проверяем первый вектор: модель возвращает одинаковую длину для всех.
        Дешевле упасть здесь с понятным текстом, чем ловить ошибку приведения
        типа на стороне PostgreSQL.
        """

        if not vectors:
            return
        width = len(vectors[0])
        if width != EMBEDDING_DIM:
            raise RuntimeError(
                f"модель вернула вектор длиной {width}, "
                f"а схема рассчитана на {EMBEDDING_DIM}. "
                "Либо смените модель, либо пересоздайте колонку embedding."
            )


def main() -> int:
    """Проверка: качаем модель, считаем пару векторов, смотрим на близость."""

    import argparse

    parser = argparse.ArgumentParser(description="Проверка эмбеддера")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    print(f"модель: {args.model}")
    print("загружаю (при первом запуске скачается ~2 ГБ)...")
    emb = Embedder(args.model, threads=args.threads)

    docs = [
        "Настройка PostgreSQL: создание роли, базы и расширения pgvector.",
        "Рецепт борща: свёкла, капуста, мясо, варить два часа.",
        "Подключение к серверу 10.17.1.51 по RDP через xfreerdp.",
    ]
    print("считаю вектора для трёх документов...")
    vectors = emb.embed_passages(docs)
    print(f"размерность: {len(vectors[0])}")

    query = "как поставить pgvector в postgres"
    qv = emb.embed_query(query)

    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    print(f"\nзапрос: {query!r}")
    scored = sorted(
        ((cosine(qv, v), d) for v, d in zip(vectors, docs)), reverse=True
    )
    for score, doc in scored:
        print(f"  {score:.4f}  {doc[:60]}")

    if scored[0][1] is not docs[0]:
        print("\nВНИМАНИЕ: ближайшим оказался не релевантный документ")
        return 1
    print("\nближайший документ — релевантный, порядок верный")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
