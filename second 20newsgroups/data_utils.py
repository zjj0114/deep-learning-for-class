from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split


CATEGORIES = ["alt.atheism", "soc.religion.christian"]
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


@dataclass
class SplitData:
    texts: list[str]
    labels: list[int]


@dataclass
class PreparedData:
    train: SplitData
    val: SplitData
    test: SplitData
    word_to_idx: dict[str, int]
    idx_to_word: list[str]
    label_names: list[str]


def strip_headers(raw_text: str) -> str:
    parts = re.split(r"\n\s*\n", raw_text, maxsplit=1)
    return parts[1] if len(parts) == 2 else raw_text


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_category_folder(folder: Path) -> list[str]:
    texts: list[str] = []
    for file_path in sorted(folder.iterdir()):
        if file_path.is_file():
            raw = file_path.read_text(encoding="latin1", errors="ignore")
            cleaned = preprocess_text(raw)
            if cleaned:
                texts.append(cleaned)
    return texts


def load_from_local_root(root_dir: Path) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    labels: list[int] = []

    train_dir = root_dir / "20news-bydate-train"
    test_dir = root_dir / "20news-bydate-test"
    if train_dir.exists() and test_dir.exists():
        for label, category in enumerate(CATEGORIES):
            for base_dir in (train_dir / category, test_dir / category):
                docs = read_category_folder(base_dir)
                texts.extend(docs)
                labels.extend([label] * len(docs))
        return texts, labels

    merged_dir = root_dir / "20_newsgroups"
    if merged_dir.exists():
        for label, category in enumerate(CATEGORIES):
            docs = read_category_folder(merged_dir / category)
            texts.extend(docs)
            labels.extend([label] * len(docs))
        return texts, labels

    if all((root_dir / category).exists() for category in CATEGORIES):
        for label, category in enumerate(CATEGORIES):
            docs = read_category_folder(root_dir / category)
            texts.extend(docs)
            labels.extend([label] * len(docs))
        return texts, labels

    raise FileNotFoundError(f"Cannot locate 20 Newsgroups folders under: {root_dir}")


def load_from_bydate_root(root_dir: Path) -> tuple[SplitData, SplitData]:
    train_dir = root_dir / "20news-bydate-train"
    test_dir = root_dir / "20news-bydate-test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Cannot locate bydate train/test folders under: {root_dir}")

    train_texts: list[str] = []
    train_labels: list[int] = []
    test_texts: list[str] = []
    test_labels: list[int] = []

    for label, category in enumerate(CATEGORIES):
        train_docs = read_category_folder(train_dir / category)
        test_docs = read_category_folder(test_dir / category)
        train_texts.extend(train_docs)
        train_labels.extend([label] * len(train_docs))
        test_texts.extend(test_docs)
        test_labels.extend([label] * len(test_docs))

    return SplitData(train_texts, train_labels), SplitData(test_texts, test_labels)


def load_binary_20news(
    data_root: Path | None = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    min_freq: int = 2,
    max_vocab_size: int = 30000,
    seed: int = 42,
) -> PreparedData:
    project_root = Path(__file__).resolve().parents[1]
    sklearn_cache_roots = [
        project_root / "data" / "sklearn_data",
        project_root.parent / ".tmp" / "sklearn_data",
    ]
    for sklearn_data_home in sklearn_cache_roots:
        if not sklearn_data_home.exists():
            continue
        try:
            newsgroups_train = fetch_20newsgroups(
                subset="train",
                categories=CATEGORIES,
                remove=("headers", "footers", "quotes"),
                data_home=str(sklearn_data_home),
                download_if_missing=False,
            )
            newsgroups_test = fetch_20newsgroups(
                subset="test",
                categories=CATEGORIES,
                remove=("headers", "footers", "quotes"),
                data_home=str(sklearn_data_home),
                download_if_missing=False,
            )

            train_texts_full = [preprocess_text(doc) for doc in newsgroups_train.data]
            test_texts = [preprocess_text(doc) for doc in newsgroups_test.data]
            train_labels_full = newsgroups_train.target.tolist()
            test_labels = newsgroups_test.target.tolist()

            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_texts_full,
                train_labels_full,
                test_size=val_size,
                stratify=train_labels_full,
                random_state=seed,
            )

            word_to_idx, idx_to_word = build_vocab(
                train_texts_full + test_texts,
                min_freq=min_freq,
                max_vocab_size=max_vocab_size,
            )
            return PreparedData(
                train=SplitData(train_texts, train_labels),
                val=SplitData(val_texts, val_labels),
                test=SplitData(test_texts, test_labels),
                word_to_idx=word_to_idx,
                idx_to_word=idx_to_word,
                label_names=CATEGORIES,
            )
        except Exception:
            pass

    candidate_roots = []
    if data_root is not None:
        candidate_roots.append(data_root)
    candidate_roots.extend(
        [
            project_root / "data",
            project_root / "data" / "20news_data",
            project_root.parent / ".tmp" / "20news_data",
            project_root.parent / ".tmp" / "sklearn_data",
        ]
    )

    texts: list[str] | None = None
    labels: list[int] | None = None
    last_error: Exception | None = None
    for root in candidate_roots:
        if not root.exists():
            continue
        try:
            train_split, test_split = load_from_bydate_root(root)
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_split.texts,
                train_split.labels,
                test_size=val_size,
                stratify=train_split.labels,
                random_state=seed,
            )
            word_to_idx, idx_to_word = build_vocab(
                train_texts,
                min_freq=min_freq,
                max_vocab_size=max_vocab_size,
            )
            return PreparedData(
                train=SplitData(train_texts, train_labels),
                val=SplitData(val_texts, val_labels),
                test=test_split,
                word_to_idx=word_to_idx,
                idx_to_word=idx_to_word,
                label_names=CATEGORIES,
            )
        except Exception as exc:
            last_error = exc
        try:
            texts, labels = load_from_local_root(root)
            break
        except Exception as exc:
            last_error = exc

    if texts is None or labels is None:
        msg = "Local 20 Newsgroups data not found."
        if last_error is not None:
            msg += f" Last error: {last_error}"
        raise FileNotFoundError(msg)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts,
        train_labels,
        test_size=val_size,
        stratify=train_labels,
        random_state=seed,
    )

    word_to_idx, idx_to_word = build_vocab(train_texts, min_freq=min_freq, max_vocab_size=max_vocab_size)
    return PreparedData(
        train=SplitData(train_texts, train_labels),
        val=SplitData(val_texts, val_labels),
        test=SplitData(test_texts, test_labels),
        word_to_idx=word_to_idx,
        idx_to_word=idx_to_word,
        label_names=CATEGORIES,
    )


def build_vocab(
    texts: list[str],
    min_freq: int = 2,
    max_vocab_size: int = 30000,
) -> tuple[dict[str, int], list[str]]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(text.split())

    idx_to_word = [PAD_TOKEN, UNK_TOKEN]
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if len(idx_to_word) >= max_vocab_size:
            break
        idx_to_word.append(token)

    word_to_idx = {word: idx for idx, word in enumerate(idx_to_word)}
    return word_to_idx, idx_to_word


def encode_text(text: str, word_to_idx: dict[str, int], max_len: int) -> list[int]:
    unk_idx = word_to_idx[UNK_TOKEN]
    tokens = text.split()[:max_len]
    return [word_to_idx.get(token, unk_idx) for token in tokens]
