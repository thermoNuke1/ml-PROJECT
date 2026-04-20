"""Train RNN/GRU/LSTM on NPZ shards produced by extract_rnn_game_shards_parallel.py.

Usage examples (Windows):

    python scripts/train_simplernn_from_shards_torch.py ^
        --input-dir data/rnn_shards_250000 ^
        --output data/torch_rnn_landmark_board_250000.json ^
        --model-type rnn --device cuda

    python scripts/train_simplernn_from_shards_torch.py ^
        --input-dir data/rnn_shards_250000 ^
        --output data/torch_gru_midweight_250000.json ^
        --model-type gru --dropout 0.2 ^
        --midgame-weight-start 20 --midgame-weight-end 50 ^
        --midgame-weight-factor 1.5 --device cuda

    python scripts/train_simplernn_from_shards_torch.py ^
        --input-dir data/rnn_shards_250000 ^
        --output data/torch_rnn_landmark_board_250000.json ^
        --checkpoint-dir data/torch_rnn_landmark_board_250000_ckpts ^
        --resume-from data/torch_rnn_landmark_board_250000_ckpts/latest.pt
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, log_loss
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, IterableDataset


CLASS_ORDER = ["white_win", "black_win", "draw"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PyTorch SimpleRNN from sharded NPZ game sequences."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--move-interval", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dense-units", type=int, default=32)
    parser.add_argument("--model-type", choices=["rnn", "gru", "lstm"], default="rnn")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--midgame-weight-start", type=int, default=None)
    parser.add_argument("--midgame-weight-end", type=int, default=None)
    parser.add_argument("--midgame-weight-factor", type=float, default=1.0)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for periodic training checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-every-epochs",
        type=int,
        default=1,
        help="Save a checkpoint every N completed epochs.",
    )
    parser.add_argument(
        "--report-every-steps",
        type=int,
        default=200,
        help="Print training progress every N optimizer steps.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional checkpoint path to resume training from.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_shards(split_dir: Path) -> list[Path]:
    return sorted(split_dir.glob("shard_*.npz"))


def count_examples(shards: list[Path], move_interval: int) -> int:
    total = 0
    for shard in shards:
        data = np.load(shard)
        offsets = data["offsets"]
        for idx in range(len(offsets) - 1):
            seq_len = int(offsets[idx + 1] - offsets[idx])
            total += seq_len // (move_interval * 2)
    return total


class LandmarkShardDataset(IterableDataset):
    def __init__(self, shards: list[Path], move_interval: int, shuffle_shards: bool):
        self.shards = shards
        self.move_interval = move_interval
        self.shuffle_shards = shuffle_shards

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        shards = self.shards
        if worker_info is not None:
            shards = shards[worker_info.id :: worker_info.num_workers]

        shards = list(shards)
        if self.shuffle_shards:
            random.shuffle(shards)

        for shard in shards:
            data = np.load(shard)
            sequences = data["sequences"]
            offsets = data["offsets"]
            targets = data["targets"]

            game_indices = list(range(len(targets)))
            if self.shuffle_shards:
                random.shuffle(game_indices)

            for idx in game_indices:
                start = int(offsets[idx])
                end = int(offsets[idx + 1])
                seq = sequences[start:end]
                label = int(targets[idx])
                max_full_move = seq.shape[0] // 2

                landmarks = list(range(self.move_interval, max_full_move + 1, self.move_interval))
                if self.shuffle_shards:
                    random.shuffle(landmarks)

                for full_move in landmarks:
                    prefix_len = full_move * 2
                    prefix = torch.from_numpy(seq[:prefix_len].copy())
                    yield prefix, label, full_move


def collate_batch(batch):
    sequences, labels, landmarks = zip(*batch)
    lengths = torch.tensor([seq.shape[0] for seq in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    landmarks_tensor = torch.tensor(landmarks, dtype=torch.long)
    return padded, lengths, labels_tensor, landmarks_tensor


class SequenceClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        dense_units: int,
        model_type: str,
        dropout: float,
    ):
        super().__init__()
        if model_type == "gru":
            self.rnn = nn.GRU(input_dim, hidden_size, batch_first=True)
        elif model_type == "lstm":
            self.rnn = nn.LSTM(input_dim, hidden_size, batch_first=True)
        else:
            self.rnn = nn.RNN(input_dim, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, dense_units)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_units, len(CLASS_ORDER))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _packed_out, hidden = self.rnn(packed)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        last_hidden = hidden[-1]
        out = self.fc1(last_hidden)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def make_loader(
    shards: list[Path],
    move_interval: int,
    batch_size: int,
    shuffle_shards: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = LandmarkShardDataset(shards, move_interval, shuffle_shards)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=collate_batch,
    )


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
):
    model.eval()
    all_probs = []
    all_labels = []
    all_landmarks = []
    with torch.no_grad():
        for xb, lengths, yb, landmarks in loader:
            xb = xb.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(xb, lengths)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(yb.numpy())
            all_landmarks.append(landmarks.numpy())

    proba = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    landmarks = np.concatenate(all_landmarks, axis=0)
    predictions = proba.argmax(axis=1)
    return proba, labels, landmarks, predictions


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


def save_training_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_val_loss: float,
    patience_left: int,
    history: dict[str, list[float] | int | None],
    args: argparse.Namespace,
    ) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_loss": float(best_val_loss),
            "patience_left": int(patience_left),
            "history": history,
            "args": vars(args),
        },
        checkpoint_path,
    )


def build_sample_weights(landmarks: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    weights = torch.ones_like(landmarks, dtype=torch.float32)
    if (
        args.midgame_weight_start is not None
        and args.midgame_weight_end is not None
        and args.midgame_weight_factor != 1.0
    ):
        mask = (landmarks >= args.midgame_weight_start) & (landmarks <= args.midgame_weight_end)
        weights = torch.where(
            mask,
            torch.full_like(weights, float(args.midgame_weight_factor)),
            weights,
        )
    return weights


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    use_amp = device.type == "cuda"
    pin_memory = device.type == "cuda"

    manifest = json.loads((args.input_dir / "manifest.json").read_text(encoding="utf-8"))
    train_shards = list_shards(args.input_dir / "train")
    val_shards = list_shards(args.input_dir / "val")
    test_shards = list_shards(args.input_dir / "test")

    sample = np.load(train_shards[0])
    input_dim = int(sample["sequences"].shape[1])

    train_loader = make_loader(
        train_shards,
        args.move_interval,
        args.batch_size,
        shuffle_shards=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    eval_workers = 0 if args.num_workers == 0 else min(2, args.num_workers)

    val_loader = make_loader(
        val_shards,
        args.move_interval,
        args.batch_size,
        shuffle_shards=False,
        num_workers=eval_workers,
        pin_memory=pin_memory,
    )
    test_loader = make_loader(
        test_shards,
        args.move_interval,
        args.batch_size,
        shuffle_shards=False,
        num_workers=eval_workers,
        pin_memory=pin_memory,
    )

    train_examples = count_examples(train_shards, args.move_interval)
    val_examples = count_examples(val_shards, args.move_interval)
    test_examples = count_examples(test_shards, args.move_interval)
    train_steps = math.ceil(train_examples / args.batch_size)

    model = SequenceClassifier(
        input_dim,
        args.hidden_size,
        args.dense_units,
        args.model_type,
        args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(reduction="none")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    best_state = None
    patience = 3
    patience_left = patience
    start_epoch = 0
    history = {
        "epochs": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "best_epoch": None,
    }

    if args.resume_from is not None:
        resume_payload = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(resume_payload["model_state_dict"])
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        scaler.load_state_dict(resume_payload["scaler_state_dict"])
        best_val_loss = float(resume_payload.get("best_val_loss", best_val_loss))
        patience_left = int(resume_payload.get("patience_left", patience_left))
        start_epoch = int(resume_payload.get("epoch", 0)) + 1
        history.update(resume_payload.get("history", {}))
        print(
            f"Resumed from checkpoint: {args.resume_from} | "
            f"starting at epoch {start_epoch + 1}/{args.epochs}",
            flush=True,
        )

    checkpoint_dir = args.checkpoint_dir
    latest_checkpoint_path = (
        checkpoint_dir / "latest.pt" if checkpoint_dir is not None else None
    )
    best_checkpoint_path = (
        checkpoint_dir / "best.pt" if checkpoint_dir is not None else None
    )

    print(
        f"Training setup | device={device} | train_sequences={train_examples} | "
        f"val_sequences={val_examples} | test_sequences={test_examples} | "
        f"train_steps_per_epoch={train_steps} | model_type={args.model_type}",
        flush=True,
    )

    for epoch_idx in range(start_epoch, args.epochs):
        model.train()
        step_count = 0
        epoch_start = time.time()
        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_examples = 0
        for xb, lengths, yb, landmarks in train_loader:
            xb = xb.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            sample_weights = build_sample_weights(landmarks, args).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(xb, lengths)
                per_example_loss = criterion(logits, yb)
                loss = (per_example_loss * sample_weights).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = int(yb.size(0))
            predictions = logits.detach().argmax(dim=1)
            epoch_loss_sum += float(loss.detach().item()) * batch_size
            epoch_correct += int((predictions == yb).sum().item())
            epoch_examples += batch_size

            step_count += 1
            if args.report_every_steps > 0 and (
                step_count == 1
                or step_count % args.report_every_steps == 0
                or step_count >= train_steps
            ):
                elapsed = time.time() - epoch_start
                steps_per_second = step_count / elapsed if elapsed > 0 else 0.0
                eta_seconds = (
                    (train_steps - step_count) / steps_per_second
                    if steps_per_second > 0
                    else 0.0
                )
                print(
                    f"Epoch {epoch_idx + 1}/{args.epochs} | "
                    f"step {step_count}/{train_steps} | "
                    f"loss={loss.item():.4f} | "
                    f"steps/s={steps_per_second:.2f} | "
                    f"eta={eta_seconds / 60.0:.1f}m",
                    flush=True,
                )
            if step_count >= train_steps:
                break

        val_proba, y_val, _val_landmarks, val_pred = evaluate_model(model, val_loader, device, use_amp)
        train_loss = epoch_loss_sum / epoch_examples if epoch_examples > 0 else 0.0
        train_accuracy = epoch_correct / epoch_examples if epoch_examples > 0 else 0.0
        val_loss = log_loss(y_val, val_proba, labels=[0, 1, 2])
        val_accuracy = accuracy_score(y_val, val_pred)
        history["epochs"].append(int(epoch_idx + 1))
        history["train_loss"].append(float(train_loss))
        history["train_accuracy"].append(float(train_accuracy))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_accuracy))

        print(
            f"Epoch {epoch_idx + 1}/{args.epochs} complete | "
            f"train_accuracy={train_accuracy:.4f} | train_loss={train_loss:.4f} | "
            f"val_accuracy={val_accuracy:.4f} | val_log_loss={val_loss:.4f}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history["best_epoch"] = int(epoch_idx + 1)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
            if best_checkpoint_path is not None:
                save_training_checkpoint(
                    best_checkpoint_path,
                    model,
                    optimizer,
                    scaler,
                    epoch_idx,
                    best_val_loss,
                    patience_left,
                    history,
                    args,
                )
        else:
            patience_left -= 1
        if latest_checkpoint_path is not None and (
            args.checkpoint_every_epochs > 0
            and (
                (epoch_idx + 1) % args.checkpoint_every_epochs == 0
                or patience_left <= 0
                or epoch_idx + 1 == args.epochs
            )
        ):
            save_training_checkpoint(
                latest_checkpoint_path,
                model,
                optimizer,
                scaler,
                epoch_idx,
                best_val_loss,
                patience_left,
                history,
                args,
            )
        if patience_left <= 0:
            print("Early stopping triggered.", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_proba, y_val, val_landmarks, val_pred = evaluate_model(model, val_loader, device, use_amp)
    test_proba, y_test, test_landmarks, test_pred = evaluate_model(model, test_loader, device, use_amp)

    output = {
        "input_dir": str(args.input_dir),
        "model": f"torch_{args.model_type}_shards",
        "architecture": {
            "model_type": args.model_type,
            "hidden_size": int(args.hidden_size),
            "dense_units": int(args.dense_units),
            "dropout": float(args.dropout),
            "optimizer": "adam",
            "learning_rate": float(args.learning_rate),
            "batch_size": int(args.batch_size),
            "max_epochs": int(args.epochs),
            "device": str(device),
            "num_workers": int(args.num_workers),
            "pin_memory": bool(pin_memory),
            "persistent_workers": bool(args.num_workers > 0),
            "automatic_mixed_precision": bool(use_amp),
            "midgame_weight_start": args.midgame_weight_start,
            "midgame_weight_end": args.midgame_weight_end,
            "midgame_weight_factor": float(args.midgame_weight_factor),
        },
        "move_interval": int(args.move_interval),
        "train_games": int(manifest["splits"]["train"]["games"]),
        "val_games": int(manifest["splits"]["val"]["games"]),
        "test_games": int(manifest["splits"]["test"]["games"]),
        "train_sequences": int(train_examples),
        "val_sequences": int(val_examples),
        "test_sequences": int(test_examples),
        "feature_count": int(input_dim),
        "training_history": history,
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
