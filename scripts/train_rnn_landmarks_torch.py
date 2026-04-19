"""Train a landmark-based RNN in PyTorch, optionally on GPU.

This mirrors the TensorFlow-based `train_rnn_landmarks.py` workflow but uses
PyTorch so CUDA can be used on Windows in this environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
    parser = argparse.ArgumentParser(
        description="Train a PyTorch landmark-based RNN on chess sequences."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rnn-type", choices=["rnn", "gru", "lstm"], default="rnn")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dense-units", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--move-interval", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_games(frame: pd.DataFrame, test_size: float, val_size: float):
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
    enriched = frame.copy()
    enriched["mover_is_white"] = (enriched["mover"] == "white").astype(int)
    enriched["side_to_move_is_white"] = (enriched["side_to_move"] == "white").astype(int)
    return enriched


def fit_feature_scaler(train_frame: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_frame[SEQUENCE_FEATURES])
    return scaler


def build_sequence_examples(
    frame: pd.DataFrame,
    game_ids: set[str],
    scaler: StandardScaler,
    move_interval: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int32),
        )

    max_len = max(sequence.shape[0] for sequence in groups)
    x = np.zeros((len(groups), max_len, len(SEQUENCE_FEATURES)), dtype=np.float32)
    lengths = np.zeros((len(groups),), dtype=np.int64)
    for index, sequence in enumerate(groups):
        x[index, : sequence.shape[0], :] = sequence
        lengths[index] = sequence.shape[0]

    y = np.asarray(labels, dtype=np.int64)
    lm = np.asarray(landmarks, dtype=np.int32)
    return x, y, lm, lengths


class SequenceClassifier(nn.Module):
    def __init__(self, input_dim: int, rnn_type: str, hidden_size: int, dense_units: int):
        super().__init__()
        if rnn_type == "rnn":
            self.rnn = nn.RNN(input_dim, hidden_size, batch_first=True)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(input_dim, hidden_size, batch_first=True)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(input_dim, hidden_size, batch_first=True)
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        self.head = nn.Sequential(
            nn.Linear(hidden_size, dense_units),
            nn.ReLU(),
            nn.Linear(dense_units, len(CLASS_ORDER)),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, hidden = self.rnn(packed)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        last_hidden = hidden[-1]
        return self.head(last_hidden)


def make_loader(x: np.ndarray, y: np.ndarray, lengths: np.ndarray, batch_size: int, shuffle: bool):
    dataset = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(lengths),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def predict_proba(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for xb, _yb, lengths in loader:
            xb = xb.to(device)
            lengths = lengths.to(device)
            logits = model(xb, lengths)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            outputs.append(probs)
    return np.concatenate(outputs, axis=0)


def multiclass_brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    truth = np.zeros((len(y_true), len(CLASS_ORDER)), dtype=np.float32)
    truth[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((proba - truth) ** 2, axis=1)))


def evaluate_landmarks(y_true: np.ndarray, proba: np.ndarray, landmarks: np.ndarray):
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
                "log_loss": float(log_loss(y_true[mask], proba[mask], labels=[0, 1, 2])),
                "brier_score": multiclass_brier_score(y_true[mask], proba[mask]),
            }
        )
    return results


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    frame = pd.read_csv(args.input)
    frame = frame.dropna(subset=["game_id", "result", "ply_index"]).copy()
    frame["target"] = prepare_target(frame)
    frame = frame.dropna(subset=["target"]).copy()
    frame = add_binary_columns(frame)

    train_games, val_games, test_games = split_games(frame, args.test_size, args.val_size)
    scaler = fit_feature_scaler(frame[frame["game_id"].isin(train_games)])

    x_train, y_train, lm_train, len_train = build_sequence_examples(frame, train_games, scaler, args.move_interval)
    x_val, y_val, lm_val, len_val = build_sequence_examples(frame, val_games, scaler, args.move_interval)
    x_test, y_test, lm_test, len_test = build_sequence_examples(frame, test_games, scaler, args.move_interval)

    train_loader = make_loader(x_train, y_train, len_train, args.batch_size, True)
    val_loader = make_loader(x_val, y_val, len_val, args.batch_size, False)
    test_loader = make_loader(x_test, y_test, len_test, args.batch_size, False)

    model = SequenceClassifier(
        input_dim=x_train.shape[2],
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        dense_units=args.dense_units,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    patience = 3
    patience_left = patience

    for _epoch in range(args.epochs):
        model.train()
        for xb, yb, lengths in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            logits = model(xb, lengths)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        val_proba = predict_proba(model, val_loader, device)
        val_loss = log_loss(y_val, val_proba, labels=[0, 1, 2])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_proba = predict_proba(model, val_loader, device)
    test_proba = predict_proba(model, test_loader, device)
    val_pred = val_proba.argmax(axis=1)
    test_pred = test_proba.argmax(axis=1)

    output = {
        "input": str(args.input),
        "model": "torch_rnn",
        "architecture": {
            "type": args.rnn_type,
            "hidden_size": int(args.hidden_size),
            "dense_units": int(args.dense_units),
            "optimizer": "adam",
            "learning_rate": float(args.learning_rate),
            "batch_size": int(args.batch_size),
            "max_epochs": int(args.epochs),
            "device": str(device),
        },
        "move_interval": int(args.move_interval),
        "train_games": int(len(train_games)),
        "val_games": int(len(val_games)),
        "test_games": int(len(test_games)),
        "train_sequences": int(len(y_train)),
        "val_sequences": int(len(y_val)),
        "test_sequences": int(len(y_test)),
        "feature_count": int(x_train.shape[2]),
        "val_accuracy": float(accuracy_score(y_val, val_pred)),
        "val_log_loss": float(log_loss(y_val, val_proba, labels=[0, 1, 2])),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_log_loss": float(log_loss(y_test, test_proba, labels=[0, 1, 2])),
        "landmarks": evaluate_landmarks(y_test, test_proba, lm_test),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"Torch RNN metrics written to: {args.output}")


if __name__ == "__main__":
    main()
