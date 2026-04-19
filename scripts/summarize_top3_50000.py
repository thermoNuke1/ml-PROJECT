"""Summarize and plot the top-3 model comparison on 50k games.

This helper script reads the saved JSON metrics for the three selected 50k
models, writes a compact CSV summary table, and generates a single landmark
accuracy plot for the report.

Run:
    python scripts/summarize_top3_50000.py

Inputs:
    Hard-coded JSON files in `data/` for:
    - board logistic regression (50k)
    - basic SimpleRNN (50k)
    - tuned GRU (50k)

Outputs:
    - data/top3_50000_summary.csv
    - data/top3_50000_landmark_accuracy.png
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


DATA_DIR = Path("data")
SUMMARY_CSV = DATA_DIR / "top3_50000_summary.csv"
PLOT_PATH = DATA_DIR / "top3_50000_landmark_accuracy.png"

MODELS = [
    {
        "label": "Board Logistic Regression (50k)",
        "metrics_path": DATA_DIR / "dev_board_logreg_metrics_50000_games.json",
        "landmarks_path": DATA_DIR / "landmark_board_logreg_50000.json",
    },
    {
        "label": "Basic SimpleRNN (50k)",
        "metrics_path": DATA_DIR / "rnn_landmark_board_50000.json",
        "landmarks_path": DATA_DIR / "rnn_landmark_board_50000.json",
    },
    {
        "label": "Tuned GRU (50k)",
        "metrics_path": DATA_DIR / "rnn_gru_dropout_midweight_50000.json",
        "landmarks_path": DATA_DIR / "rnn_gru_dropout_midweight_50000.json",
    },
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def landmark_accuracy(landmarks: list[dict], move: int):
    for row in landmarks:
        if int(row["full_move_landmark"]) == move:
            return float(row["accuracy"])
    return None


def write_summary(rows: list[dict]):
    fieldnames = [
        "model",
        "test_accuracy",
        "test_log_loss",
        "avg_landmark_accuracy",
        "avg_landmark_brier",
        "move_5_accuracy",
        "move_10_accuracy",
        "move_20_accuracy",
        "move_30_accuracy",
        "move_40_accuracy",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(model_payloads: list[tuple[str, list[dict]]]):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 7))

    for label, landmarks in model_payloads:
        xs = [int(row["full_move_landmark"]) for row in landmarks if int(row["full_move_landmark"]) <= 40]
        ys = [float(row["accuracy"]) for row in landmarks if int(row["full_move_landmark"]) <= 40]
        ax.plot(xs, ys, marker="o", linewidth=2.4, markersize=5, label=label)

    ax.set_title("Top 3 Models on 50k Games: Landmark Accuracy", fontsize=15, pad=12)
    ax.set_xlabel("Full Move Landmark", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xticks(list(range(5, 41, 5)))
    ax.set_ylim(0.50, 0.76)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=220, bbox_inches="tight")


def main():
    summary_rows = []
    plot_payloads = []

    for model in MODELS:
        metrics = load_json(model["metrics_path"])
        landmarks_payload = load_json(model["landmarks_path"])
        landmarks = landmarks_payload["landmarks"]

        summary_rows.append(
            {
                "model": model["label"],
                "test_accuracy": f"{float(metrics['test_accuracy']):.6f}",
                "test_log_loss": f"{float(metrics['test_log_loss']):.6f}",
                "avg_landmark_accuracy": f"{average([float(row['accuracy']) for row in landmarks]):.6f}",
                "avg_landmark_brier": f"{average([float(row['brier_score']) for row in landmarks]):.6f}",
                "move_5_accuracy": f"{landmark_accuracy(landmarks, 5):.6f}",
                "move_10_accuracy": f"{landmark_accuracy(landmarks, 10):.6f}",
                "move_20_accuracy": f"{landmark_accuracy(landmarks, 20):.6f}",
                "move_30_accuracy": f"{landmark_accuracy(landmarks, 30):.6f}",
                "move_40_accuracy": f"{landmark_accuracy(landmarks, 40):.6f}",
            }
        )
        plot_payloads.append((model["label"], landmarks))

    write_summary(summary_rows)
    plot_curves(plot_payloads)
    print(f"Wrote summary to {SUMMARY_CSV}")
    print(f"Saved plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
