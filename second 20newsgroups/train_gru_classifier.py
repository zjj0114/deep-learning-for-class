from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from data_utils import CATEGORIES, PreparedData, encode_text, load_binary_20news


@dataclass
class TrainConfig:
    model_type: str
    data_root: str | None
    seed: int
    max_len: int
    min_freq: int
    max_vocab_size: int
    batch_size: int
    embedding_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float
    lr: float
    weight_decay: float
    epochs: int
    patience: int


class NewsDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], word_to_idx: dict[str, int], max_len: int) -> None:
        self.encoded = [encode_text(text, word_to_idx, max_len) for text in texts]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self.encoded[index], self.labels[index]


def collate_batch(batch: list[tuple[list[int], int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences, labels = zip(*batch)
    lengths = torch.tensor([max(1, len(seq)) for seq in sequences], dtype=torch.long)
    max_batch_len = int(lengths.max().item())
    padded = torch.zeros(len(sequences), max_batch_len, dtype=torch.long)

    for row, seq in enumerate(sequences):
        if seq:
            padded[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)

    return padded, lengths, torch.tensor(labels, dtype=torch.long)


class SequenceClassifier(nn.Module):
    def __init__(
        self,
        model_type: str,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.model_type = model_type.lower()
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM, "rnn": nn.RNN}[self.model_type]
        self.encoder = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=recurrent_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.encoder(packed)
        if self.model_type == "lstm":
            hidden = hidden[0]
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        mask = torch.arange(output.size(1), device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1)

        output_masked = output.masked_fill(~mask, -1e9)
        max_pool = output_masked.max(dim=1).values

        output_sum = (output * mask).sum(dim=1)
        mean_pool = output_sum / lengths.unsqueeze(1)

        hidden = hidden.view(self.encoder.num_layers, 2, tokens.size(0), self.encoder.hidden_size)
        last_hidden = torch.cat([hidden[-1, 0], hidden[-1, 1]], dim=1)

        features = torch.cat([last_hidden, mean_pool + max_pool], dim=1)
        return self.classifier(self.dropout(features))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    all_labels: list[int] = []
    all_preds: list[int] = []

    with torch.no_grad():
        for tokens, lengths, labels in loader:
            tokens = tokens.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            logits = model(tokens, lengths)
            loss = criterion(logits, labels)

            total_loss += float(loss.detach().item()) * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    avg_loss = total_loss / max(1, len(loader.dataset))
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_labels, all_preds


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    all_labels: list[int] = []
    all_preds: list[int] = []

    for tokens, lengths, labels in loader:
        tokens = tokens.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.detach().item()) * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    avg_loss = total_loss / max(1, len(loader.dataset))
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def plot_history(history: dict[str, list[float]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CATEGORIES)
    ax.set_yticks([0, 1], CATEGORIES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BiGRU classifier on 20 Newsgroups binary data.")
    parser.add_argument("--model-type", type=str, default="gru", choices=["gru", "lstm", "rnn"])
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-len", type=int, default=400)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-vocab-size", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts",
    )
    return parser.parse_args()


def save_outputs(
    output_dir: Path,
    config: TrainConfig,
    prepared: PreparedData,
    history: dict[str, list[float]],
    test_loss: float,
    test_acc: float,
    y_true: list[int],
    y_pred: list[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_history(history, output_dir / "training_history.png")

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion(cm, output_dir / "confusion_matrix.png")

    report = classification_report(y_true, y_pred, target_names=prepared.label_names, digits=4)
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    metrics = {
        "config": asdict(config),
        "dataset": {
            "train_size": len(prepared.train.labels),
            "val_size": len(prepared.val.labels),
            "test_size": len(prepared.test.labels),
            "vocab_size": len(prepared.word_to_idx),
            "labels": prepared.label_names,
        },
        "best_val_acc": max(history["val_acc"]),
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "confusion_matrix": cm.tolist(),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        model_type=args.model_type,
        data_root=args.data_root,
        seed=args.seed,
        max_len=args.max_len,
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
    )

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepared = load_binary_20news(
        data_root=Path(config.data_root) if config.data_root else None,
        min_freq=config.min_freq,
        max_vocab_size=config.max_vocab_size,
        seed=config.seed,
    )

    train_ds = NewsDataset(prepared.train.texts, prepared.train.labels, prepared.word_to_idx, config.max_len)
    val_ds = NewsDataset(prepared.val.texts, prepared.val.labels, prepared.word_to_idx, config.max_len)
    test_ds = NewsDataset(prepared.test.texts, prepared.test.labels, prepared.word_to_idx, config.max_len)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch)

    model = SequenceClassifier(
        model_type=config.model_type,
        vocab_size=len(prepared.word_to_idx),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0

    print(f"Using device: {device}")
    print(f"Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print(f"Vocab size: {len(prepared.word_to_idx)}")

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= config.patience:
                print("Early stopping triggered.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    model.load_state_dict(best_state)
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "word_to_idx": prepared.word_to_idx,
            "label_names": prepared.label_names,
            "config": asdict(config),
        },
        output_dir / "best_model.pt",
    )

    save_outputs(output_dir, config, prepared, history, test_loss, test_acc, y_true, y_pred)

    print()
    print("Test result")
    print(f"  test_loss     : {test_loss:.4f}")
    print(f"  test_accuracy : {test_acc:.4f}")
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
