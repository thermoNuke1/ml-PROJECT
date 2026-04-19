"""Evaluate chess result models at fixed full-move landmarks.

This script trains a model using the same pipeline as train_baseline_model.py
and then evaluates it on one snapshot per test game at fixed full-move
intervals. For move k, the script uses the board position after Black's kth
move, which corresponds to ply index 2*k.

The purpose is to avoid overweighting long games and to make model quality
easier to interpret at different stages of the game.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from train_baseline_model import (
    build_model,
    get_available_feature_columns,
    load_dataset,
    oversample_draw_rows,
    prepare_target,
    split_by_game,
)


CLASS_ORDER = ["white_win", "black_win", "draw"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for landmark evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model at fixed 5-move landmarks."
    )
    parser.add_argument("--input", required=True, type=Path, help="Feature CSV path.")
    parser.add_argument(
        "--model",
        required=True,
        choices=["logreg", "hgbt", "mlp", "mlp_deep", "mlp_wide", "mlp_balanced", "svm"],
        help="Model family to evaluate.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of games reserved for the test split.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Fraction of remaining non-test games reserved for validation.",
    )
    parser.add_argument(
        "--oversample-draw-factor",
        type=float,
        default=1.0,
        help="Training-only oversampling multiplier for draw rows.",
    )
    parser.add_argument(
        "--move-interval",
        type=int,
        default=5,
        help="Evaluate every N full moves. Defaults to 5.",
    )
    parser.add_argument(
        "--min-games-per-landmark",
        type=int,
        default=30,
        help="Minimum number of test games required to report a landmark.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination JSON file for landmark metrics.",
    )
    return parser.parse_args()


def multiclass_brier_score(y_true: pd.Series, proba: np.ndarray, labels: list[str]) -> float:
    """Compute the multiclass Brier score."""
    true_matrix = np.zeros((len(y_true), len(labels)), dtype=float)
    label_to_index = {label: index for index, label in enumerate(labels)}

    for row_index, label in enumerate(y_true):
        true_matrix[row_index, label_to_index[label]] = 1.0

    return float(np.mean(np.sum((proba - true_matrix) ** 2, axis=1)))


def evaluate_landmarks(
    test_frame: pd.DataFrame,
    selected_features: list[str],
    model,
    move_interval: int,
    min_games_per_landmark: int,
) -> list[dict[str, object]]:
    """Evaluate one trained model at fixed full-move landmarks."""
    max_full_moves = int(test_frame["ply_index"].max() // 2)
    landmark_results: list[dict[str, object]] = []

    for full_move in range(move_interval, max_full_moves + 1, move_interval):
        target_ply = full_move * 2
        landmark_frame = test_frame[test_frame["ply_index"] == target_ply].copy()

        if len(landmark_frame) < min_games_per_landmark:
            continue

        x_landmark = landmark_frame[selected_features]
        y_landmark = landmark_frame["target"]
        y_pred = model.predict(x_landmark)
        y_proba_raw = model.predict_proba(x_landmark)

        aligned = np.zeros((len(y_landmark), len(CLASS_ORDER)), dtype=float)
        class_index = {label: index for index, label in enumerate(model.classes_)}
        for label_index, label in enumerate(CLASS_ORDER):
            if label in class_index:
                aligned[:, label_index] = y_proba_raw[:, class_index[label]]

        landmark_results.append(
            {
                "full_move_landmark": int(full_move),
                "games": int(landmark_frame["game_id"].nunique()),
                "rows": int(len(landmark_frame)),
                "accuracy": float(accuracy_score(y_landmark, y_pred)),
                "log_loss": float(log_loss(y_landmark, aligned, labels=CLASS_ORDER)),
                "brier_score": multiclass_brier_score(
                    y_true=y_landmark,
                    proba=aligned,
                    labels=CLASS_ORDER,
                ),
            }
        )

    return landmark_results


def main() -> None:
    """Train a model and evaluate it at 5-move landmarks."""
    args = parse_args()
    frame = load_dataset(args.input, max_rows=None)
    frame = frame.dropna(subset=["game_id", "result", "ply_index"]).copy()
    frame["target"] = prepare_target(frame)
    frame = frame.dropna(subset=["target"]).copy()

    numeric_features, categorical_features = get_available_feature_columns(frame)
    selected_features = numeric_features + categorical_features

    train_frame, val_frame, test_frame = split_by_game(
        frame=frame,
        test_size=args.test_size,
        val_size=args.val_size,
    )

    x_train = train_frame[selected_features]
    y_train = train_frame["target"]
    x_train, y_train = oversample_draw_rows(
        x_train,
        y_train,
        factor=args.oversample_draw_factor,
    )

    model = build_model(args.model, numeric_features, categorical_features)
    model.fit(x_train, y_train)

    landmark_results = evaluate_landmarks(
        test_frame=test_frame,
        selected_features=selected_features,
        model=model,
        move_interval=args.move_interval,
        min_games_per_landmark=args.min_games_per_landmark,
    )

    output = {
        "input": str(args.input),
        "model": args.model,
        "oversample_draw_factor": float(args.oversample_draw_factor),
        "move_interval": int(args.move_interval),
        "train_games": int(train_frame["game_id"].nunique()),
        "val_games": int(val_frame["game_id"].nunique()),
        "test_games": int(test_frame["game_id"].nunique()),
        "numeric_feature_count": int(len(numeric_features)),
        "categorical_feature_count": int(len(categorical_features)),
        "landmarks": landmark_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(json.dumps(output, indent=2))
    print(f"Landmark metrics written to: {args.output}")


if __name__ == "__main__":
    main()
