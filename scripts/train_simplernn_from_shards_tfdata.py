"""Train a SimpleRNN from sharded NPZ game sequences via tf.data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, log_loss


CLASS_ORDER = ["white_win", "black_win", "draw"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a TensorFlow SimpleRNN from sharded NPZ game sequences."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--move-interval", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dense-units", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    return parser.parse_args()


def list_shards(split_dir: Path) -> list[Path]:
    return sorted(split_dir.glob("shard_*.npz"))


def count_examples_in_split(shards: list[Path], move_interval: int) -> int:
    total = 0
    for shard in shards:
        data = np.load(shard)
        offsets = data["offsets"]
        for idx in range(len(offsets) - 1):
            seq_len = offsets[idx + 1] - offsets[idx]
            total += seq_len // (move_interval * 2)
    return total


def shard_generator(shards: list[Path], move_interval: int):
    for shard in shards:
        data = np.load(shard)
        sequences = data["sequences"]
        offsets = data["offsets"]
        targets = data["targets"]

        for idx in range(len(targets)):
            start = int(offsets[idx])
            end = int(offsets[idx + 1])
            sequence = sequences[start:end]
            label = int(targets[idx])
            max_full_move = sequence.shape[0] // 2
            for full_move in range(move_interval, max_full_move + 1, move_interval):
                prefix_len = full_move * 2
                yield sequence[:prefix_len].astype(np.float32), np.int32(label), np.int32(full_move)


def build_dataset(shards: list[Path], move_interval: int, batch_size: int, training: bool):
    output_signature = (
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: shard_generator(shards, move_interval),
        output_signature=output_signature,
    )
    if training:
        dataset = dataset.shuffle(20000, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=([None, None], [], []),
        padding_values=(0.0, np.int32(0), np.int32(0)),
    )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def build_model(input_dim: int, hidden_size: int, dense_units: int, learning_rate: float) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(None, input_dim))
    masked = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    hidden = tf.keras.layers.SimpleRNN(hidden_size)(masked)
    dense = tf.keras.layers.Dense(dense_units, activation="relu")(hidden)
    outputs = tf.keras.layers.Dense(len(CLASS_ORDER), activation="softmax")(dense)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def collect_predictions(model: tf.keras.Model, dataset: tf.data.Dataset):
    all_proba = []
    all_labels = []
    all_landmarks = []
    for xb, yb, landmarks in dataset:
        proba = model.predict_on_batch(xb)
        all_proba.append(proba)
        all_labels.append(yb.numpy())
        all_landmarks.append(landmarks.numpy())
    return (
        np.concatenate(all_proba, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_landmarks, axis=0),
    )


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
    train_shards = list_shards(args.input_dir / "train")
    val_shards = list_shards(args.input_dir / "val")
    test_shards = list_shards(args.input_dir / "test")

    sample_shard = np.load(train_shards[0])
    input_dim = sample_shard["sequences"].shape[1]

    train_dataset = build_dataset(train_shards, args.move_interval, args.batch_size, training=True)
    val_dataset = build_dataset(val_shards, args.move_interval, args.batch_size, training=False)
    test_dataset = build_dataset(test_shards, args.move_interval, args.batch_size, training=False)

    train_examples = count_examples_in_split(train_shards, args.move_interval)
    val_examples = count_examples_in_split(val_shards, args.move_interval)
    test_examples = count_examples_in_split(test_shards, args.move_interval)
    train_steps = int(np.ceil(train_examples / args.batch_size))
    val_steps = int(np.ceil(val_examples / args.batch_size))

    model = build_model(input_dim, args.hidden_size, args.dense_units, args.learning_rate)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]
    fit_history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=2,
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

    val_proba, y_val, val_landmarks = collect_predictions(model, val_dataset)
    test_proba, y_test, test_landmarks = collect_predictions(model, test_dataset)
    val_pred = val_proba.argmax(axis=1)
    test_pred = test_proba.argmax(axis=1)

    manifest = json.loads((args.input_dir / "manifest.json").read_text(encoding="utf-8"))
    output = {
        "input_dir": str(args.input_dir),
        "model": "tfdata_simplernn",
        "architecture": {
            "hidden_size": int(args.hidden_size),
            "dense_units": int(args.dense_units),
            "optimizer": "adam",
            "learning_rate": float(args.learning_rate),
            "batch_size": int(args.batch_size),
            "max_epochs": int(args.epochs),
        },
        "move_interval": int(args.move_interval),
        "train_games": int(manifest["splits"]["train"]["games"]),
        "val_games": int(manifest["splits"]["val"]["games"]),
        "test_games": int(manifest["splits"]["test"]["games"]),
        "train_sequences": int(train_examples),
        "val_sequences": int(val_examples),
        "test_sequences": int(test_examples),
        "feature_count": int(input_dim),
        "training_history": history_payload,
        "val_accuracy": float(accuracy_score(y_val, val_pred)),
        "val_log_loss": float(log_loss(y_val, val_proba, labels=[0, 1, 2])),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_log_loss": float(log_loss(y_test, test_proba, labels=[0, 1, 2])),
        "landmarks": evaluate_landmarks(y_test, test_proba, test_landmarks),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
