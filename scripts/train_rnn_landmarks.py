"""Train landmark-based recurrent models from flat board-aware CSV features.

This is the earlier TensorFlow/Keras recurrent trainer used before the NPZ
shard pipeline was added. It groups move-level CSV rows by game, builds prefix
examples at fixed move landmarks, and trains a recurrent model.

Example:
    python scripts/train_rnn_landmarks.py ^
        --input data/dev_board_features_1000_games.csv ^
        --rnn-type simplernn ^
        --metrics-output data/rnn_landmark_board_1000.json

Try a GRU or LSTM:
    python scripts/train_rnn_landmarks.py ^
        --input data/dev_board_features_1000_games.csv ^
        --rnn-type gru ^
        --hidden-units 64 ^
        --metrics-output data/rnn_gru_64_1000.json

Important flags:
    --input:
        Board-aware CSV produced by `extract_lichess_board_features.py`.
    --rnn-type:
        `simplernn`, `gru`, or `lstm`.
    --move-interval:
        Full-move interval used to define landmarks.
    --hidden-units:
        Recurrent hidden size.
    --metrics-output:
        Destination JSON file for test metrics and landmark curves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from train_baseline_model import prepare_target


CLASS_ORDER = ["white_win", "black_win", "draw"]
SEQUENCE_FEATURES = [
    "white_elo",
    "black_elo",
    "elo_diff_white_minus_black",
    "is_capture",
    "is_check",
    "is_checkmate",
    "is_castle",
    "is_promotion",
    "san_length",
    "white_time_seconds",
    "black_time_seconds",
    "mover_time_seconds",
    "opponent_time_seconds",
    "mover_time_spent_seconds",
    "white_time_ratio",
    "black_time_ratio",
    "clock_diff_seconds_white_minus_black",
    "legal_moves_count",
    "halfmove_clock",
    "white_material",
    "black_material",
    "material_diff_white_minus_black",
    "white_pawns",
    "black_pawns",
    "white_knights",
    "black_knights",
    "white_bishops",
    "black_bishops",
    "white_rooks",
    "black_rooks",
    "white_queens",
    "black_queens",
    "white_has_bishop_pair",
    "black_has_bishop_pair",
    "white_can_castle_kingside",
    "white_can_castle_queenside",
    "black_can_castle_kingside",
    "black_can_castle_queenside",
    "is_insufficient_material",
    "mover_is_white",
    "side_to_move_is_white",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for RNN training."""
    parser = argparse.ArgumentParser(
        description="Train a basic landmark-based RNN on chess sequences."
    )
    parser.add_argument("--input", required=True, type=Path, help="Feature CSV path.")
    parser.add_argument(
        "--rnn-type",
        choices=["simplernn", "gru", "lstm"],
        default="simplernn",
        help="Type of recurrent layer to use.",
    )
    parser.add_argument(
        "--hidden-layers",
        default="64",
        help="Comma-separated recurrent hidden sizes, e.g. '64' or '64,64'.",
    )
    parser.add_argument(
        "--dense-units",
        type=int,
        default=32,
        help="Dense layer width after the recurrent stack.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Input dropout rate for recurrent layers.",
    )
    parser.add_argument(
        "--recurrent-dropout",
        type=float,
        default=0.0,
        help="Recurrent dropout rate for recurrent layers.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size.",
    )
    parser.add_argument(
        "--move-interval",
        type=int,
        default=5,
        help="Use prefixes ending every N full moves.",
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
        help="Fraction of remaining games reserved for validation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--landmark-weight-start",
        type=int,
        default=None,
        help="Optional full-move landmark at which to start upweighting examples.",
    )
    parser.add_argument(
        "--landmark-weight-end",
        type=int,
        default=None,
        help="Optional full-move landmark at which to stop upweighting examples.",
    )
    parser.add_argument(
        "--landmark-weight-factor",
        type=float,
        default=1.0,
        help="Multiplier applied to landmarks inside the chosen range.",
    )
    parser.add_argument(
        "--draw-weight-factor",
        type=float,
        default=1.0,
        help="Additional training weight for draw examples.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination JSON file for metrics.",
    )
    return parser.parse_args()


def split_games(frame: pd.DataFrame, test_size: float, val_size: float):
    """Split game ids into train/validation/test partitions."""
    game_targets = (
        frame[["game_id", "target"]]
        .drop_duplicates(subset=["game_id"])
        .dropna(subset=["game_id", "target"])
    )
    stratify_labels = None
    counts = game_targets["target"].value_counts()
    if not counts.empty and int(counts.min()) >= 2:
        stratify_labels = game_targets["target"]

    train_val_games, test_games = train_test_split(
        game_targets["game_id"],
        test_size=test_size,
        random_state=42,
        shuffle=True,
        stratify=stratify_labels,
    )

    remaining = game_targets[game_targets["game_id"].isin(train_val_games)]
    remaining_stratify = None
    counts = remaining["target"].value_counts()
    if not counts.empty and int(counts.min()) >= 2:
        remaining_stratify = remaining["target"]

    train_games, val_games = train_test_split(
        remaining["game_id"],
        test_size=val_size,
        random_state=42,
        shuffle=True,
        stratify=remaining_stratify,
    )
    return set(train_games), set(val_games), set(test_games)


def add_binary_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add simple binary categorical encodings for sequence models."""
    enriched = frame.copy()
    enriched["mover_is_white"] = (enriched["mover"] == "white").astype(int)
    enriched["side_to_move_is_white"] = (
        enriched["side_to_move"] == "white"
    ).astype(int)
    return enriched


def fit_feature_scaler(train_frame: pd.DataFrame) -> StandardScaler:
    """Fit a scaler on training rows only."""
    scaler = StandardScaler()
    scaler.fit(train_frame[SEQUENCE_FEATURES])
    return scaler


def build_sequence_examples(
    frame: pd.DataFrame,
    game_ids: set[str],
    scaler: StandardScaler,
    move_interval: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build padded prefix sequences ending at 5-move landmarks."""
    groups = []
    labels = []
    landmarks = []

    subset = frame[frame["game_id"].isin(game_ids)].copy()
    for _game_id, game_frame in subset.groupby("game_id", sort=False):
        game_frame = game_frame.sort_values("ply_index").copy()
        max_full_move = int(game_frame["ply_index"].max() // 2)
        target_label = game_frame["target"].iloc[0]

        for full_move in range(move_interval, max_full_move + 1, move_interval):
            target_ply = full_move * 2
            prefix = game_frame[game_frame["ply_index"] <= target_ply].copy()
            if prefix.empty or prefix["ply_index"].max() != target_ply:
                continue

            features = scaler.transform(prefix[SEQUENCE_FEATURES])
            groups.append(features.astype(np.float32))
            labels.append(CLASS_ORDER.index(target_label))
            landmarks.append(full_move)

    if not groups:
        return (
            np.zeros((0, 1, len(SEQUENCE_FEATURES)), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
        )

    max_len = max(sequence.shape[0] for sequence in groups)
    x = np.zeros((len(groups), max_len, len(SEQUENCE_FEATURES)), dtype=np.float32)
    for index, sequence in enumerate(groups):
        x[index, : sequence.shape[0], :] = sequence

    y = np.asarray(labels, dtype=np.int32)
    lm = np.asarray(landmarks, dtype=np.int32)
    return x, y, lm


def build_model(
    input_dim: int,
    rnn_type: str,
    hidden_layers: list[int],
    dense_units: int,
    dropout: float,
    recurrent_dropout: float,
    learning_rate: float,
) -> tf.keras.Model:
    """Create a configurable recurrent model."""
    inputs = tf.keras.Input(shape=(None, input_dim))
    masked = tf.keras.layers.Masking(mask_value=0.0)(inputs)

    if rnn_type == "simplernn":
        layer_cls = tf.keras.layers.SimpleRNN
    elif rnn_type == "gru":
        layer_cls = tf.keras.layers.GRU
    elif rnn_type == "lstm":
        layer_cls = tf.keras.layers.LSTM
    else:
        raise ValueError(f"Unsupported rnn_type: {rnn_type}")

    hidden = masked
    for index, units in enumerate(hidden_layers):
        return_sequences = index < len(hidden_layers) - 1
        hidden = layer_cls(
            units,
            return_sequences=return_sequences,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
        )(hidden)

    dense = tf.keras.layers.Dense(dense_units, activation="relu")(hidden)
    outputs = tf.keras.layers.Dense(len(CLASS_ORDER), activation="softmax")(dense)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_sample_weights(
    labels: np.ndarray,
    landmarks: np.ndarray,
    landmark_weight_start: int | None,
    landmark_weight_end: int | None,
    landmark_weight_factor: float,
    draw_weight_factor: float,
) -> np.ndarray:
    """Create sample weights for training examples."""
    weights = np.ones(len(labels), dtype=np.float32)

    if (
        landmark_weight_start is not None
        and landmark_weight_end is not None
        and landmark_weight_factor > 1.0
    ):
        mask = (landmarks >= landmark_weight_start) & (landmarks <= landmark_weight_end)
        weights[mask] *= landmark_weight_factor

    if draw_weight_factor > 1.0:
        draw_index = CLASS_ORDER.index("draw")
        weights[labels == draw_index] *= draw_weight_factor

    return weights


def multiclass_brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Compute the multiclass Brier score."""
    truth = np.zeros((len(y_true), len(CLASS_ORDER)), dtype=np.float32)
    truth[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((proba - truth) ** 2, axis=1)))


def evaluate_landmarks(y_true: np.ndarray, proba: np.ndarray, landmarks: np.ndarray):
    """Evaluate test predictions at each full-move landmark."""
    results = []
    predictions = proba.argmax(axis=1)

    for full_move in sorted(np.unique(landmarks)):
        mask = landmarks == full_move
        if int(mask.sum()) < 30:
            continue

        results.append(
            {
                "full_move_landmark": int(full_move),
                "games": int(mask.sum()),
                "rows": int(mask.sum()),
                "accuracy": float(accuracy_score(y_true[mask], predictions[mask])),
                "log_loss": float(
                    log_loss(
                        y_true[mask],
                        proba[mask],
                        labels=list(range(len(CLASS_ORDER))),
                    )
                ),
                "brier_score": multiclass_brier_score(y_true[mask], proba[mask]),
            }
        )
    return results


def main() -> None:
    """Train the RNN and export landmark metrics."""
    args = parse_args()
    tf.random.set_seed(42)
    np.random.seed(42)
    hidden_layers = [int(value) for value in args.hidden_layers.split(",") if value.strip()]

    frame = pd.read_csv(args.input)
    frame = frame.dropna(subset=["game_id", "result", "ply_index"]).copy()
    frame["target"] = prepare_target(frame)
    frame = frame.dropna(subset=["target"]).copy()
    frame = add_binary_columns(frame)

    train_games, val_games, test_games = split_games(
        frame,
        test_size=args.test_size,
        val_size=args.val_size,
    )

    scaler = fit_feature_scaler(frame[frame["game_id"].isin(train_games)])
    x_train, y_train, train_landmarks = build_sequence_examples(
        frame, train_games, scaler, args.move_interval
    )
    x_val, y_val, val_landmarks = build_sequence_examples(
        frame, val_games, scaler, args.move_interval
    )
    x_test, y_test, test_landmarks = build_sequence_examples(
        frame, test_games, scaler, args.move_interval
    )

    model = build_model(
        input_dim=len(SEQUENCE_FEATURES),
        rnn_type=args.rnn_type,
        hidden_layers=hidden_layers,
        dense_units=args.dense_units,
        dropout=args.dropout,
        recurrent_dropout=args.recurrent_dropout,
        learning_rate=args.learning_rate,
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        )
    ]
    sample_weights = compute_sample_weights(
        labels=y_train,
        landmarks=train_landmarks,
        landmark_weight_start=args.landmark_weight_start,
        landmark_weight_end=args.landmark_weight_end,
        landmark_weight_factor=args.landmark_weight_factor,
        draw_weight_factor=args.draw_weight_factor,
    )
    fit_history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        sample_weight=sample_weights,
        verbose=0,
        callbacks=callbacks,
    )
    history_payload = {
        "epochs": list(range(1, len(fit_history.history.get("loss", [])) + 1)),
        "train_loss": [float(value) for value in fit_history.history.get("loss", [])],
        "train_accuracy": [
            float(value) for value in fit_history.history.get("accuracy", [])
        ],
        "val_loss": [float(value) for value in fit_history.history.get("val_loss", [])],
        "val_accuracy": [
            float(value) for value in fit_history.history.get("val_accuracy", [])
        ],
    }
    if history_payload["val_loss"]:
        best_epoch_index = int(np.argmin(history_payload["val_loss"]))
        history_payload["best_epoch"] = int(history_payload["epochs"][best_epoch_index])
    else:
        history_payload["best_epoch"] = None

    val_proba = model.predict(x_val, verbose=0)
    test_proba = model.predict(x_test, verbose=0)
    val_pred = val_proba.argmax(axis=1)
    test_pred = test_proba.argmax(axis=1)

    output = {
        "input": str(args.input),
        "model": "basic_rnn",
        "architecture": {
            "type": args.rnn_type,
            "recurrent_layers": hidden_layers,
            "dense_units": int(args.dense_units),
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": float(args.learning_rate),
            "dropout": float(args.dropout),
            "recurrent_dropout": float(args.recurrent_dropout),
            "batch_size": int(args.batch_size),
            "max_epochs": int(args.epochs),
            "landmark_weight_start": args.landmark_weight_start,
            "landmark_weight_end": args.landmark_weight_end,
            "landmark_weight_factor": float(args.landmark_weight_factor),
            "draw_weight_factor": float(args.draw_weight_factor),
        },
        "move_interval": int(args.move_interval),
        "train_games": int(len(train_games)),
        "val_games": int(len(val_games)),
        "test_games": int(len(test_games)),
        "train_sequences": int(len(x_train)),
        "val_sequences": int(len(x_val)),
        "test_sequences": int(len(x_test)),
        "feature_count": int(len(SEQUENCE_FEATURES)),
        "training_history": history_payload,
        "val_accuracy": float(accuracy_score(y_val, val_pred)),
        "val_log_loss": float(
            log_loss(y_val, val_proba, labels=list(range(len(CLASS_ORDER))))
        ),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_log_loss": float(
            log_loss(y_test, test_proba, labels=list(range(len(CLASS_ORDER))))
        ),
        "landmarks": evaluate_landmarks(y_test, test_proba, test_landmarks),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"RNN metrics written to: {args.output}")


if __name__ == "__main__":
    main()
