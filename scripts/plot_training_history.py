"""Plot training/validation loss curves from saved trainer JSON outputs.

Examples:
    python scripts/plot_training_history.py ^
        --input data/rnn_landmark_board_10000_with_history.json ^
        --output data/rnn_landmark_board_10000_loss_curve.png ^
        --title "SimpleRNN Loss Curves (10,000 games)"

    python scripts/plot_training_history.py ^
        --input data/rnn_landmark_board_10000_with_history.json ^
        --input data/rnn_landmark_board_50000_with_history.json ^
        --input data/rnn_landmark_board_100000_with_history.json ^
        --input data/torch_rnn_landmark_board_250000_with_history.json ^
        --output data/rnn_loss_curves_10k_to_250k.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot training and validation loss curves from trainer JSON outputs."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=Path,
        help="Path to a trainer JSON file containing training_history.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination PNG path.",
    )
    parser.add_argument(
        "--title",
        default="Training vs Validation Loss",
        help="Optional plot title.",
    )
    return parser.parse_args()


def load_payload(path: Path) -> tuple[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    history = payload.get("training_history")
    if not history:
        raise ValueError(f"{path} does not contain training_history.")

    label = path.stem.replace("_with_history", "")
    train_games = payload.get("train_games")
    if train_games is not None:
        label = f"{label} ({train_games:,} train games)"
    return label, history


def main() -> None:
    args = parse_args()
    fig, ax = plt.subplots(figsize=(11, 7))

    for input_path in args.input:
        label, history = load_payload(input_path)
        epochs = history["epochs"]
        ax.plot(
            epochs,
            history["train_loss"],
            marker="o",
            linewidth=2.0,
            label=f"{label} train",
        )
        ax.plot(
            epochs,
            history["val_loss"],
            marker="s",
            linewidth=2.0,
            linestyle="--",
            label=f"{label} val",
        )

    ax.set_title(args.title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
