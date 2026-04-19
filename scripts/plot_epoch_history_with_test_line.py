"""Plot epoch-wise training/validation loss with a final test-loss reference line.

This is useful when the run stores validation history by epoch, but only a final
test loss after training finishes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot epoch loss history with a final test-loss reference line."
    )
    parser.add_argument("--input", required=True, type=Path, help="Trainer JSON path.")
    parser.add_argument("--output", required=True, type=Path, help="PNG output path.")
    parser.add_argument("--title", required=True, help="Plot title.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    history = payload["training_history"]
    epochs = history["epochs"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    test_loss = float(payload["test_log_loss"])
    best_epoch = history.get("best_epoch")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(epochs, train_loss, marker="o", linewidth=2.2, label="Training loss")
    ax.plot(epochs, val_loss, marker="s", linewidth=2.2, label="Validation loss")
    ax.axhline(
        test_loss,
        color="black",
        linestyle=":",
        linewidth=2.0,
        label=f"Final test loss = {test_loss:.4f}",
    )
    if best_epoch is not None:
        ax.axvline(
            int(best_epoch),
            color="gray",
            linestyle="--",
            linewidth=1.5,
            label=f"Best epoch = {best_epoch}",
        )

    ax.set_title(args.title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Log loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
