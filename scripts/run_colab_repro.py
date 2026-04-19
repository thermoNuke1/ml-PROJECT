"""Run a lightweight reproducibility path for Google Colab.

This script is meant to give the teaching team a quick, executable path that
works in Colab without requiring the full multi-gigabyte preprocessing runs.
It uses the tracked sample PGN to regenerate small development artifacts and
prints the archived larger-scale metrics that support the report tables.

Example:
    python scripts/run_colab_repro.py

    python scripts/run_colab_repro.py --run-rnn-smoke --rnn-epochs 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = REPO_ROOT / "artifacts" / "reported_results"
SAMPLE_PGN = REPO_ROOT / "artifacts" / "sample_data" / "colab_dev_60_games.pgn"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Colab reproducibility runner."""
    parser = argparse.ArgumentParser(
        description="Run the lightweight Google Colab reproducibility path."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "colab_runs",
        help="Directory where regenerated sample outputs will be written.",
    )
    parser.add_argument(
        "--run-rnn-smoke",
        action="store_true",
        help="Also run a tiny PyTorch shard-based smoke test on the sample PGN.",
    )
    parser.add_argument(
        "--rnn-epochs",
        type=int,
        default=2,
        help="Number of epochs for the optional RNN smoke test.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    """Run a subprocess command from the repo root and fail loudly if needed.

    Args:
        command: Command and arguments to execute.
    """
    print(f"\n[run] {' '.join(command)}")
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def load_json(path: Path) -> dict:
    """Load a UTF-8 JSON file.

    Args:
        path: JSON file path.

    Returns:
        dict: Parsed payload.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def print_archived_metrics() -> None:
    """Print the tracked report metrics archived for Colab access."""
    board_metrics = load_json(ARCHIVE_DIR / "dev_board_logreg_metrics_1000_games.json")
    rnn_metrics = load_json(ARCHIVE_DIR / "torch_rnn_landmark_board_250000_with_history.json")

    print("\nArchived report metrics")
    print(
        "Board logistic regression (1k games): "
        f"test_accuracy={board_metrics['test_accuracy']:.3f}, "
        f"test_log_loss={board_metrics['test_log_loss']:.3f}"
    )
    print(
        "Torch SimpleRNN from 250k-game shards: "
        f"test_accuracy={rnn_metrics['test_accuracy']:.3f}, "
        f"test_log_loss={rnn_metrics['test_log_loss']:.3f}"
    )


def run_sample_board_pipeline(output_dir: Path) -> None:
    """Regenerate a small board-feature baseline from the tracked sample PGN.

    Args:
        output_dir: Destination directory for regenerated artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    board_features_path = output_dir / "sample_board_features.csv"
    metrics_path = output_dir / "sample_board_logreg_metrics.json"

    run_command(
        [
            sys.executable,
            "scripts/extract_lichess_board_features.py",
            "--input",
            str(SAMPLE_PGN),
            "--output",
            str(board_features_path),
            "--workers",
            "1",
        ]
    )
    run_command(
        [
            sys.executable,
            "scripts/train_baseline_model.py",
            "--input",
            str(board_features_path),
            "--model",
            "logreg",
            "--metrics-output",
            str(metrics_path),
        ]
    )

    metrics = load_json(metrics_path)
    print("\nRegenerated sample baseline")
    print(
        f"rows(train/val/test)=({metrics['train_rows']}/"
        f"{metrics['val_rows']}/{metrics['test_rows']})"
    )
    print(
        "sample board logreg: "
        f"test_accuracy={metrics['test_accuracy']:.3f}, "
        f"test_log_loss={metrics['test_log_loss']:.3f}"
    )


def run_sample_rnn_smoke(output_dir: Path, epochs: int) -> None:
    """Run a tiny shard extraction and PyTorch RNN smoke test on sample data.

    Args:
        output_dir: Destination directory for smoke-test artifacts.
        epochs: Number of training epochs.
    """
    shard_dir = output_dir / "sample_rnn_shards"
    metrics_path = output_dir / "sample_torch_rnn_metrics.json"

    run_command(
        [
            sys.executable,
            "scripts/extract_rnn_game_shards_parallel.py",
            "--input",
            str(SAMPLE_PGN),
            "--output-dir",
            str(shard_dir),
            "--workers",
            "1",
            "--chunk-games",
            "4",
            "--games-per-shard",
            "32",
        ]
    )
    run_command(
        [
            sys.executable,
            "scripts/train_simplernn_from_shards_torch.py",
            "--input-dir",
            str(shard_dir),
            "--output",
            str(metrics_path),
            "--epochs",
            str(epochs),
            "--batch-size",
            "16",
            "--hidden-size",
            "32",
            "--dense-units",
            "16",
            "--num-workers",
            "0",
            "--device",
            "cpu",
        ]
    )

    metrics = load_json(metrics_path)
    print("\nRegenerated sample RNN smoke test")
    print(
        "sample torch rnn: "
        f"test_accuracy={metrics['test_accuracy']:.3f}, "
        f"test_log_loss={metrics['test_log_loss']:.3f}"
    )


def main() -> None:
    """Run the quick Colab reproducibility flow."""
    args = parse_args()
    if not SAMPLE_PGN.exists():
        raise FileNotFoundError(f"Missing sample PGN: {SAMPLE_PGN}")
    if not ARCHIVE_DIR.exists():
        raise FileNotFoundError(f"Missing archived metrics directory: {ARCHIVE_DIR}")

    print_archived_metrics()
    run_sample_board_pipeline(args.output_dir)

    if args.run_rnn_smoke:
        run_sample_rnn_smoke(args.output_dir, args.rnn_epochs)

    print(f"\nOutputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
