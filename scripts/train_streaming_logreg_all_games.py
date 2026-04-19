"""Train a streaming logistic-style classifier directly from the filtered PGN.

This script is designed for the "all games" setting where creating a giant
intermediate CSV would be too expensive. It reuses the board-aware feature
construction logic from `extract_lichess_board_features.py`, but instead of
writing rows to disk it hashes each sampled move position into a sparse vector
and updates an out-of-core `SGDClassifier(loss="log_loss")`.

To keep the full-data pass practical, the default behavior samples one
landmark snapshot every 5 full moves rather than every single ply. This still
uses all games while reducing the total number of training examples
substantially.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import chess
import chess.pgn
import joblib
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier

from extract_lichess_board_features import build_row, parse_time_control_seconds


CLASS_ORDER = np.array(["white_win", "black_win", "draw"], dtype=object)
RESULT_TO_LABEL = {
    "1-0": "white_win",
    "0-1": "black_win",
    "1/2-1/2": "draw",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a streaming logistic-style model from the full filtered PGN."
    )
    parser.add_argument("--input", required=True, type=Path, help="Filtered PGN path.")
    parser.add_argument(
        "--model-output",
        required=True,
        type=Path,
        help="Destination .joblib path for the trained streaming model bundle.",
    )
    parser.add_argument(
        "--stats-output",
        required=True,
        type=Path,
        help="Destination JSON path for training statistics.",
    )
    parser.add_argument(
        "--move-interval",
        type=int,
        default=5,
        help="Use one training example every N full moves. Default: 5.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20000,
        help="Number of sampled positions per partial_fit batch.",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=2**18,
        help="Hashed feature dimension. Default: 262144.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-6,
        help="L2 regularization strength for SGDClassifier.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Optional cap for benchmarking/debugging.",
    )
    parser.add_argument(
        "--report-every-games",
        type=int,
        default=50000,
        help="Print progress every N processed games.",
    )
    return parser.parse_args()


def stable_game_key(headers: chess.pgn.Headers, processed_games: int) -> str:
    site = headers.get("Site", "").strip()
    if site:
        return site
    white = headers.get("White", "").strip()
    black = headers.get("Black", "").strip()
    date = headers.get("Date", "").strip()
    return f"{date}|{white}|{black}|{processed_games}"


def scale_feature(name: str, value: object) -> float | None:
    if value is None:
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if name in {"white_elo", "black_elo"}:
        return numeric / 3000.0
    if name == "elo_diff_white_minus_black":
        return numeric / 1000.0
    if name == "ply_index":
        return numeric / 100.0
    if name == "san_length":
        return numeric / 20.0
    if name in {
        "white_time_seconds",
        "black_time_seconds",
        "mover_time_seconds",
        "opponent_time_seconds",
        "mover_time_spent_seconds",
        "clock_diff_seconds_white_minus_black",
    }:
        return numeric / 600.0
    if name in {"white_time_ratio", "black_time_ratio"}:
        return numeric
    if name == "legal_moves_count":
        return numeric / 100.0
    if name == "halfmove_clock":
        return numeric / 100.0
    if name in {"white_material", "black_material", "material_diff_white_minus_black"}:
        return numeric / 40.0
    if name.endswith(
        (
            "_pawns",
            "_knights",
            "_bishops",
            "_rooks",
            "_queens",
        )
    ):
        return numeric / 10.0
    return numeric


def row_to_feature_dict(row: dict[str, object]) -> dict[str, float]:
    feature_dict: dict[str, float] = {}

    for name, value in row.items():
        if name in {
            "game_id",
            "date",
            "white_player",
            "black_player",
            "result",
            "white_win",
            "black_win",
            "draw",
            "time_control",
            "termination",
            "fullmove_number",
            "san",
            "uci",
        }:
            continue

        if name in {"mover", "side_to_move"}:
            feature_dict[f"{name}={value}"] = 1.0
            continue

        scaled = scale_feature(name, value)
        if scaled is not None:
            feature_dict[name] = scaled

    return feature_dict


def flush_batch(
    batch_features: list[dict[str, float]],
    batch_labels: list[str],
    classifier: SGDClassifier,
    hasher: FeatureHasher,
    is_first_batch: bool,
) -> bool:
    if not batch_features:
        return is_first_batch

    x_batch = hasher.transform(batch_features)
    y_batch = np.asarray(batch_labels, dtype=object)

    if is_first_batch:
        classifier.partial_fit(x_batch, y_batch, classes=CLASS_ORDER)
        return False

    classifier.partial_fit(x_batch, y_batch)
    return is_first_batch


def main() -> None:
    args = parse_args()
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)

    hasher = FeatureHasher(
        n_features=args.n_features,
        input_type="dict",
        alternate_sign=False,
    )
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=args.alpha,
        learning_rate="optimal",
        average=True,
        fit_intercept=True,
        random_state=42,
    )

    batch_features: list[dict[str, float]] = []
    batch_labels: list[str] = []
    first_batch = True

    start_time = time.time()
    processed_games = 0
    sampled_positions = 0

    with args.input.open("r", encoding="utf-8", errors="replace") as pgn_handle:
        while True:
            game = chess.pgn.read_game(pgn_handle)
            if game is None:
                break

            processed_games += 1
            headers = game.headers
            game_key = stable_game_key(headers, processed_games)
            _game_hash = hashlib.md5(game_key.encode("utf-8")).hexdigest()

            result = headers.get("Result", "*")
            target_label = RESULT_TO_LABEL.get(result)
            if target_label is None:
                if args.max_games is not None and processed_games >= args.max_games:
                    break
                continue

            initial_time = parse_time_control_seconds(headers.get("TimeControl", ""))
            white_time_seconds = initial_time
            black_time_seconds = initial_time
            previous_white_time = initial_time
            previous_black_time = initial_time

            for node in game.mainline():
                mover = "white" if node.parent.board().turn == chess.WHITE else "black"
                clock_value = node.clock()

                if clock_value is not None:
                    if mover == "white":
                        previous_white_time = white_time_seconds
                        white_time_seconds = int(clock_value)
                    else:
                        previous_black_time = black_time_seconds
                        black_time_seconds = int(clock_value)

                row = build_row(
                    node=node,
                    headers=headers,
                    white_time_seconds=white_time_seconds,
                    black_time_seconds=black_time_seconds,
                    previous_white_time=previous_white_time,
                    previous_black_time=previous_black_time,
                    initial_time=initial_time,
                )

                if args.move_interval > 1:
                    if row["ply_index"] % (args.move_interval * 2) != 0:
                        continue

                batch_features.append(row_to_feature_dict(row))
                batch_labels.append(target_label)
                sampled_positions += 1

                if len(batch_features) >= args.batch_size:
                    first_batch = flush_batch(
                        batch_features=batch_features,
                        batch_labels=batch_labels,
                        classifier=classifier,
                        hasher=hasher,
                        is_first_batch=first_batch,
                    )
                    batch_features.clear()
                    batch_labels.clear()

            if processed_games % args.report_every_games == 0:
                elapsed = time.time() - start_time
                rate = sampled_positions / elapsed if elapsed > 0 else 0.0
                print(
                    f"Processed games: {processed_games} | "
                    f"Sampled positions: {sampled_positions} | "
                    f"Rows/sec: {rate:.1f}"
                )

            if args.max_games is not None and processed_games >= args.max_games:
                break

    first_batch = flush_batch(
        batch_features=batch_features,
        batch_labels=batch_labels,
        classifier=classifier,
        hasher=hasher,
        is_first_batch=first_batch,
    )

    elapsed = time.time() - start_time
    model_bundle = {
        "classifier": classifier,
        "n_features": args.n_features,
        "move_interval": args.move_interval,
        "class_order": CLASS_ORDER.tolist(),
        "feature_encoding": "FeatureHasher(dict, alternate_sign=False)",
        "scaling_notes": {
            "elo": "/3000",
            "elo_diff": "/1000",
            "ply_index": "/100",
            "clock_features_seconds": "/600",
            "legal_moves_count": "/100",
            "material_features": "/40",
            "piece_counts": "/10",
        },
    }
    joblib.dump(model_bundle, args.model_output)

    stats = {
        "input": str(args.input),
        "model_output": str(args.model_output),
        "move_interval": int(args.move_interval),
        "n_features": int(args.n_features),
        "alpha": float(args.alpha),
        "processed_games": int(processed_games),
        "sampled_positions": int(sampled_positions),
        "elapsed_seconds": float(elapsed),
        "positions_per_second": float(sampled_positions / elapsed if elapsed > 0 else 0.0),
    }
    args.stats_output.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps(stats, indent=2))
    print(f"Saved model bundle to: {args.model_output}")
    print(f"Saved training stats to: {args.stats_output}")


if __name__ == "__main__":
    main()
