"""Plot one combined landmark-accuracy figure across saved model runs.

This helper script scans the JSON metrics files listed in `LABELS`, extracts
the saved landmark accuracies, and overlays them in one figure for the report.

Run:
    python scripts/plot_landmark_accuracies.py

Inputs:
    Hard-coded metric JSON files in `data/` whose names appear in `LABELS`.

Output:
    - data/landmark_accuracy_all_models.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt


DATA_DIR = Path("data")
OUTPUT_PATH = DATA_DIR / "landmark_accuracy_all_models.png"


LABELS = {
    "landmark_light_logreg_1000.json": "Light LogReg (1k)",
    "landmark_light_hgbt_1000.json": "Light HGBT (1k)",
    "landmark_board_logreg_1000.json": "Board LogReg (1k)",
    "landmark_board_logreg_10000.json": "Board LogReg (10k)",
    "landmark_board_hgbt_1000.json": "Board HGBT (1k)",
    "landmark_board_svm_1000.json": "Board SVM (1k)",
    "landmark_board_svm_10000.json": "Board SVM (10k)",
    "landmark_board_mlp_1000.json": "Board MLP (1k)",
    "landmark_board_mlp_deep_1000.json": "Board Deep MLP (1k)",
    "landmark_board_mlp_wide_1000.json": "Board Wide MLP (1k)",
    "landmark_board_mlp_balanced_1000.json": "Board Balanced MLP (1k)",
    "rnn_simplernn_64_1000.json": "SimpleRNN (1k)",
    "rnn_gru_64_1000.json": "GRU (1k)",
    "rnn_lstm_64_1000.json": "LSTM (1k)",
    "rnn_gru_stacked_64_1000.json": "Stacked GRU (1k)",
    "rnn_gru_dropout_1000.json": "GRU + Dropout (1k)",
    "rnn_gru_tuned_1000.json": "GRU Tuned LR (1k)",
    "rnn_gru_midweight_1000.json": "GRU Midgame Weighted (1k)",
    "rnn_gru_drawweight_1000.json": "GRU Draw Weighted (1k)",
    "rnn_gru_combo_tuned_1000.json": "GRU Combo Tuned (1k)",
    "rnn_gru_dropout_midweight_1000.json": "Best Tuned GRU (1k)",
    "rnn_landmark_board_1000.json": "Basic SimpleRNN Landmark (1k)",
    "rnn_landmark_board_10000.json": "Basic SimpleRNN Landmark (10k)",
    "rnn_gru_dropout_midweight_10000.json": "Best Tuned GRU (10k)",
}


def load_series(path: Path):
    payload = json.loads(path.read_text())
    landmarks = payload.get("landmarks", [])
    xs = []
    ys = []
    for entry in landmarks:
        move = entry.get("full_move_landmark")
        acc = entry.get("accuracy")
        if move is None or acc is None or move > 40:
            continue
        xs.append(move)
        ys.append(acc)
    return xs, ys


def main():
    json_paths = sorted(
        [
            path
            for path in DATA_DIR.glob("*.json")
            if path.name in LABELS and "landmarks" in path.read_text(encoding="utf-8", errors="ignore")
        ],
        key=lambda path: LABELS[path.name].lower(),
    )

    if not json_paths:
        raise SystemExit("No landmark JSON files found to plot.")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(18, 10))

    for path in json_paths:
        xs, ys = load_series(path)
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=1.8,
            markersize=4,
            alpha=0.9,
            label=LABELS[path.name],
        )

    ax.set_title("Landmark Accuracy By Full Move Across All Saved Models", fontsize=16, pad=14)
    ax.set_xlabel("Full Move Landmark", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xticks(list(range(5, 41, 5)))
    ax.set_ylim(0.35, 0.85)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=9,
        ncol=1,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
