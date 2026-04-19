"""Plot final validation vs test loss across SimpleRNN scaling runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


RUNS = [
    ("10k", Path("data/rnn_landmark_board_10000_with_history.json")),
    ("50k", Path("data/rnn_landmark_board_50000_with_history.json")),
    ("100k", Path("data/rnn_landmark_board_100000_with_history.json")),
    ("250k", Path("data/torch_rnn_landmark_board_250000_with_history.json")),
]
OUTPUT_PATH = Path("data/val_vs_test_loss_10k_to_250k.png")


def load_metrics(path: Path) -> tuple[float, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["val_log_loss"]), float(payload["test_log_loss"])


def main() -> None:
    labels = []
    val_losses = []
    test_losses = []

    for label, path in RUNS:
        val_loss, test_loss = load_metrics(path)
        labels.append(label)
        val_losses.append(val_loss)
        test_losses.append(test_loss)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(labels, val_losses, marker="o", linewidth=2.2, label="Validation log loss")
    ax.plot(labels, test_losses, marker="s", linewidth=2.2, label="Test log loss")
    ax.set_title("SimpleRNN Validation vs Test Log Loss (10k to 250k)")
    ax.set_xlabel("Dataset size")
    ax.set_ylabel("Log loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200)
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
