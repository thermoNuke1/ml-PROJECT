import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


DATA_DIR = Path("data")
SUMMARY_PATH = DATA_DIR / "simplernn_scaling_summary.csv"
PLOT_PATH = DATA_DIR / "simplernn_scaling_landmark_accuracy_v2.png"

RUNS = [
    ("1k", DATA_DIR / "rnn_landmark_board_1000.json"),
    ("10k", DATA_DIR / "rnn_landmark_board_10000.json"),
    ("50k", DATA_DIR / "rnn_landmark_board_50000.json"),
    ("100k", DATA_DIR / "rnn_landmark_board_100000.json"),
    ("250k", DATA_DIR / "torch_rnn_landmark_board_250000.json"),
    ("1M", DATA_DIR / "torch_rnn_landmark_board_1000000.json"),
    ("8.63M", DATA_DIR / "torch_rnn_landmark_board_all_games.json"),
]

RUN_STYLES = {
    "1k": {"color": "#7f7f7f", "marker": "o", "linestyle": "-", "linewidth": 1.2, "zorder": 2},
    "10k": {"color": "#1f77b4", "marker": "s", "linestyle": "-", "linewidth": 1.2, "zorder": 3},
    "50k": {"color": "#2ca02c", "marker": "^", "linestyle": "-", "linewidth": 1.35, "zorder": 4},
    "100k": {"color": "#ff7f0e", "marker": "D", "linestyle": "--", "linewidth": 1.25, "zorder": 5},
    "250k": {"color": "#d62728", "marker": "*", "linestyle": "-", "linewidth": 1.5, "zorder": 6},
    "1M": {"color": "#9467bd", "marker": "P", "linestyle": "-", "linewidth": 1.5, "zorder": 7},
    "8.63M": {"color": "#8c564b", "marker": "X", "linestyle": "-", "linewidth": 1.65, "zorder": 8},
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def metric_at_move(landmarks: list[dict], move: int, key: str):
    for row in landmarks:
        if int(row["full_move_landmark"]) == move:
            return float(row[key])
    return None


def build_summary_rows():
    rows = []
    plot_payloads = []
    for label, path in RUNS:
        payload = load_json(path)
        landmarks = payload["landmarks"]
        rows.append(
            {
                "dataset": label,
                "test_accuracy": f"{float(payload['test_accuracy']):.6f}",
                "test_log_loss": f"{float(payload['test_log_loss']):.6f}",
                "avg_landmark_accuracy": f"{avg([float(row['accuracy']) for row in landmarks]):.6f}",
                "avg_landmark_brier": f"{avg([float(row['brier_score']) for row in landmarks]):.6f}",
                "move_5_accuracy": f"{metric_at_move(landmarks, 5, 'accuracy'):.6f}",
                "move_10_accuracy": f"{metric_at_move(landmarks, 10, 'accuracy'):.6f}",
                "move_20_accuracy": f"{metric_at_move(landmarks, 20, 'accuracy'):.6f}",
                "move_30_accuracy": f"{metric_at_move(landmarks, 30, 'accuracy'):.6f}",
                "move_40_accuracy": f"{metric_at_move(landmarks, 40, 'accuracy'):.6f}",
            }
        )
        plot_payloads.append((label, landmarks))
    return rows, plot_payloads


def write_summary(rows: list[dict]):
    fieldnames = [
        "dataset",
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
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_scaling(plot_payloads: list[tuple[str, list[dict]]]):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6.5))

    for label, landmarks in plot_payloads:
        xs = [int(row["full_move_landmark"]) for row in landmarks if int(row["full_move_landmark"]) <= 40]
        ys = [float(row["accuracy"]) for row in landmarks if int(row["full_move_landmark"]) <= 40]
        style = RUN_STYLES.get(label, {})
        ax.plot(
            xs,
            ys,
            label=f"SimpleRNN ({label})",
            markersize=5.5 if label in {"250k", "1M", "8.63M"} else 4.5,
            **style,
        )
        if xs and ys:
            ax.annotate(
                label,
                (xs[-1], ys[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color=style.get("color", "black"),
                fontweight="bold" if label in {"250k", "1M", "8.63M"} else "normal",
            )

    ax.set_title("SimpleRNN Scaling: Landmark Accuracy by Dataset Size", fontsize=15, pad=12)
    ax.set_xlabel("Full Move Landmark", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xticks(list(range(5, 41, 5)))
    ax.set_ylim(0.48, 0.77)
    ax.legend(frameon=True, loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=220, bbox_inches="tight")


def main():
    rows, plot_payloads = build_summary_rows()
    write_summary(rows)
    plot_scaling(plot_payloads)
    print(f"Wrote summary to {SUMMARY_PATH}")
    print(f"Saved plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
