"""Extract board-aware game sequence shards directly from PGN.

This script reads the filtered PGN file, reconstructs each game move by move,
and saves complete board-aware sequences into NPZ shards split into
train/validation/test by game. The shards can then be reused by RNN/GRU/LSTM
trainers without re-parsing the PGN.

Typical usage:
    python scripts/extract_rnn_game_shards_parallel.py ^
        --input data/lichess_rapid_10_0_completed.pgn ^
        --output-dir data/rnn_shards_250000 ^
        --max-games 250000 ^
        --workers 4 ^
        --chunk-games 500 ^
        --games-per-shard 5000

Resume an interrupted run:
    python scripts/extract_rnn_game_shards_parallel.py ^
        --input data/lichess_rapid_10_0_completed.pgn ^
        --output-dir data/rnn_shards_250000 ^
        --max-games 250000 ^
        --workers 4 ^
        --resume

Important flags:
    --input:
        Source PGN file. This should usually be the filtered rapid dataset.
    --output-dir:
        Folder where shard files and manifest.json will be written.
    --max-games:
        Maximum number of games to extract. Omit to run until EOF.
    --workers:
        Number of CPU worker processes. On Windows, smaller values can be more
        stable; on Linux/WSL, higher values are usually fine.
    --chunk-games:
        Number of games bundled into one worker task. Larger values reduce
        overhead but delay progress updates.
    --games-per-shard:
        Number of complete games written into each NPZ shard.
    --report-every-games:
        Progress-print interval.
    --checkpoint-every-games:
        How often checkpoint.json is updated for resume support.
    --resume:
        Continue from an existing checkpoint in the output directory.
"""

from __future__ import annotations

import argparse
import io
import os

import orjson
import xxhash
from concurrent.futures import ProcessPoolExecutor
from collections import deque
from pathlib import Path

import chess
import chess.pgn
import numpy as np

from extract_lichess_board_features import process_game


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


RESULT_TO_LABEL = {
    "1-0": 0,
    "0-1": 1,
    "1/2-1/2": 2,
}
# Precomputed divisors (coresponds to SEQUENCE_FEATURES)
# Single numpy broadcast replaces 41 Python function calls per move.
_SCALE_DIVISORS = np.array([
    3000.0,  # white_elo
    3000.0,  # black_elo
    1000.0,  # elo_diff_white_minus_black
    1.0,     # is_capture
    1.0,     # is_check
    1.0,     # is_checkmate
    1.0,     # is_castle
    1.0,     # is_promotion
    20.0,    # san_length
    600.0,   # white_time_seconds
    600.0,   # black_time_seconds
    600.0,   # mover_time_seconds
    600.0,   # opponent_time_seconds
    600.0,   # mover_time_spent_seconds
    1.0,     # white_time_ratio  (already 0–1)
    1.0,     # black_time_ratio  (already 0–1)
    600.0,   # clock_diff_seconds_white_minus_black
    100.0,   # legal_moves_count
    100.0,   # halfmove_clock
    40.0,    # white_material
    40.0,    # black_material
    40.0,    # material_diff_white_minus_black
    10.0,    # white_pawns
    10.0,    # black_pawns
    10.0,    # white_knights
    10.0,    # black_knights
    10.0,    # white_bishops
    10.0,    # black_bishops
    10.0,    # white_rooks
    10.0,    # black_rooks
    10.0,    # white_queens
    10.0,    # black_queens
    1.0,     # white_has_bishop_pair
    1.0,     # black_has_bishop_pair
    1.0,     # white_can_castle_kingside
    1.0,     # white_can_castle_queenside
    1.0,     # black_can_castle_kingside
    1.0,     # black_can_castle_queenside
    1.0,     # is_insufficient_material
    1.0,     # mover_is_white
    1.0,     # side_to_move_is_white
], dtype=np.float32)
assert len(_SCALE_DIVISORS) == len(SEQUENCE_FEATURES), "divisor/feature list mismatch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract parallel NPZ shards of board-aware game sequences."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--chunk-games", type=int, default=500)
    parser.add_argument("--games-per-shard", type=int, default=5000)
    parser.add_argument(
        "--report-every-games",
        type=int,
        default=100,
        help="Print progress every N processed games.",
    )
    parser.add_argument(
        "--checkpoint-every-games",
        type=int,
        default=5000,
        help="Write checkpoint every N processed games. Larger is faster.",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Use compressed NPZ output. Disabled by default for speed.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing checkpoint in output-dir if present.",
    )
    return parser.parse_args()


def stable_game_key(headers: chess.pgn.Headers, fallback_index: int) -> str:
    site = headers.get("Site", "").strip()
    if site:
        return site
    white = headers.get("White", "").strip()
    black = headers.get("Black", "").strip()
    date = headers.get("Date", "").strip()
    return f"{date}|{white}|{black}|{fallback_index}"


def split_name_for_key(game_key: str) -> str:
    # xxhash is ~10x faster than MD5
    value = xxhash.xxh64(game_key).intdigest() % 1000
    if value < 720:
        return "train"
    if value < 800:
        return "val"
    return "test"






def process_game_text(item: tuple[int, str]) -> dict[str, object] | None:
    fallback_index, game_text = item
    handle = io.StringIO(game_text)
    game = chess.pgn.read_game(handle)
    if game is None:
        return None

    headers = game.headers
    result = headers.get("Result", "*")
    if result not in RESULT_TO_LABEL:
        return None

    row_dicts = process_game(game)
    if not row_dicts:
        return None

    n_moves = len(row_dicts)
    n_features = len(SEQUENCE_FEATURES)

    # Pre-allocate a float32 matrix of shape (moves, features).
    # Zeros are the correct fallback for None values (e.g. missing clock data).
    matrix = np.zeros((n_moves, n_features), dtype=np.float32)

    for i, row in enumerate(row_dicts):
        # mover_is_white and side_to_move_is_white are now set by process_game
        # directly from the bool already computed there — no string comparison needed.
        for j, name in enumerate(SEQUENCE_FEATURES):
            v = row[name]
            if v is not None:
                matrix[i, j] = v


    # used to be 41 Python function calls per move
    # (one scale_feature() call per feature), each doing a bunch of ifs
    # _SCALE_DIVISORS is const array aligned 1:1 with SEQUENCE_FEATURES.
    matrix /= _SCALE_DIVISORS

    game_key = stable_game_key(headers, fallback_index)
    return {
        "split": split_name_for_key(game_key),
        "target": RESULT_TO_LABEL[result],
        "sequence": matrix,
    }


def process_chunk(items: list[tuple[int, str]]) -> list[dict[str, object]]:
    results = []
    for item in items:
        processed = process_game_text(item)
        if processed is not None:
            results.append(processed)
    return results


def yield_game_chunks(
    input_path: Path,
    max_games: int | None,
    chunk_games: int,
    skip_games: int = 0,
):
    processed_games = 0
    current_chunk: list[tuple[int, str]] = []
    with input_path.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            processed_games += 1
            if processed_games <= skip_games:
                if max_games is not None and processed_games >= max_games:
                    break
                continue
            current_chunk.append((processed_games, game.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=True))))
            if len(current_chunk) >= chunk_games:
                yield current_chunk, processed_games
                current_chunk = []
            if max_games is not None and processed_games >= max_games:
                break
    if current_chunk:
        yield current_chunk, processed_games


def flush_split_buffer(
    output_dir: Path,
    split_name: str,
    shard_index: int,
    buffer: list[dict[str, object]],
    compress: bool,
) -> int:
    if not buffer:
        return shard_index

    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    sequences = []
    offsets = [0]
    targets = []

    total_rows = 0
    for item in buffer:
        seq = item["sequence"]
        sequences.append(seq)
        total_rows += seq.shape[0]
        offsets.append(total_rows)
        targets.append(item["target"])

    concatenated = np.concatenate(sequences, axis=0)
    save_fn = np.savez_compressed if compress else np.savez
    save_fn(
        split_dir / f"shard_{shard_index:05d}.npz",
        sequences=concatenated,
        offsets=np.asarray(offsets, dtype=np.int64),
        targets=np.asarray(targets, dtype=np.int64),
    )
    return shard_index + 1


def save_pending_buffer(
    output_dir: Path,
    split_name: str,
    buffer: list[dict[str, object]],
    compress: bool,
) -> None:
    # Disabled on Windows due to repeated file-lock races when overwriting the
    # same pending shard file. We still checkpoint processed game counts and
    # completed shard indices, so resume will continue from the last processed
    # game and only lose the small not-yet-flushed in-memory remainder.
    return


def load_pending_buffer(output_dir: Path, split_name: str) -> list[dict[str, object]]:
    return []


def checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "checkpoint.json"


def save_checkpoint(
    output_dir: Path,
    processed_games: int,
    shard_indices: dict[str, int],
    split_game_counts: dict[str, int],
) -> None:
    payload = {
        "processed_games": int(processed_games),
        "shard_indices": {key: int(value) for key, value in shard_indices.items()},
        "split_game_counts": {key: int(value) for key, value in split_game_counts.items()},
    }
    # orjson.dumps returns bytes; write_bytes avoids an encode round-trip.
    checkpoint_path(output_dir).write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def load_checkpoint(output_dir: Path):
    path = checkpoint_path(output_dir)
    if not path.exists():
        return None
    return orjson.loads(path.read_bytes())


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(args.output_dir) if args.resume else None
    buffers = {split: [] for split in ["train", "val", "test"]}
    shard_indices = (
        checkpoint["shard_indices"]
        if checkpoint
        else {"train": 0, "val": 0, "test": 0}
    )
    split_game_counts = (
        checkpoint["split_game_counts"]
        if checkpoint
        else {"train": 0, "val": 0, "test": 0}
    )
    processed_games = checkpoint["processed_games"] if checkpoint else 0
    next_checkpoint_games = (
        processed_games + args.checkpoint_every_games
        if args.checkpoint_every_games > 0
        else None
    )
    next_report_games = (
        processed_games + args.report_every_games
        if args.report_every_games > 0
        else None
    )

    def maybe_save_checkpoint(force: bool = False) -> None:
        nonlocal next_checkpoint_games
        if args.checkpoint_every_games <= 0:
            return
        if not force and (next_checkpoint_games is None or processed_games < next_checkpoint_games):
            return
        save_checkpoint(
            output_dir=args.output_dir,
            processed_games=processed_games,
            shard_indices=shard_indices,
            split_game_counts=split_game_counts,
        )
        while next_checkpoint_games is not None and processed_games >= next_checkpoint_games:
            next_checkpoint_games += args.checkpoint_every_games

    def maybe_report_progress(force: bool = False) -> None:
        nonlocal next_report_games
        if args.report_every_games <= 0:
            return
        if not force and (next_report_games is None or processed_games < next_report_games):
            return
        print(
            f"Processed games: {processed_games} | "
            f"Saved train/val/test games: "
            f"{split_game_counts['train']}/{split_game_counts['val']}/{split_game_counts['test']} | "
            f"Completed train/val/test shards: "
            f"{shard_indices['train']}/{shard_indices['val']}/{shard_indices['test']}",
            flush=True,
        )
        while next_report_games is not None and processed_games >= next_report_games:
            next_report_games += args.report_every_games

    def consume_results(results: list[dict[str, object]]) -> None:
        nonlocal shard_indices, split_game_counts
        touched_splits: set[str] = set()
        for item in results:
            split_name = item["split"]
            buffers[split_name].append(item)
            split_game_counts[split_name] += 1
            touched_splits.add(split_name)
            if len(buffers[split_name]) >= args.games_per_shard:
                shard_indices[split_name] = flush_split_buffer(
                    output_dir=args.output_dir,
                    split_name=split_name,
                    shard_index=shard_indices[split_name],
                    buffer=buffers[split_name],
                    compress=args.compress,
                )
                buffers[split_name] = []
                touched_splits.add(split_name)

        for split_name in touched_splits:
            save_pending_buffer(
                args.output_dir,
                split_name,
                buffers[split_name],
                compress=args.compress,
            )

    if args.workers <= 1:
        for chunk, processed in yield_game_chunks(
            args.input,
            args.max_games,
            args.chunk_games,
            skip_games=processed_games,
        ):
            processed_games = processed
            results = process_chunk(chunk)
            consume_results(results)
            maybe_save_checkpoint()
            maybe_report_progress()
    else:
        inflight_limit = max(1, args.workers * 2)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            pending_futures = deque()
            for chunk, processed in yield_game_chunks(
                args.input,
                args.max_games,
                args.chunk_games,
                skip_games=processed_games,
            ):
                processed_games = processed
                pending_futures.append(executor.submit(process_chunk, chunk))
                if len(pending_futures) >= inflight_limit:
                    results = pending_futures.popleft().result()
                    consume_results(results)
                    maybe_save_checkpoint()
                    maybe_report_progress()

            while pending_futures:
                results = pending_futures.popleft().result()
                consume_results(results)
                maybe_save_checkpoint()
                maybe_report_progress()

    for split_name in ["train", "val", "test"]:
        shard_indices[split_name] = flush_split_buffer(
            output_dir=args.output_dir,
            split_name=split_name,
            shard_index=shard_indices[split_name],
            buffer=buffers[split_name],
            compress=args.compress,
        )
        buffers[split_name] = []
        save_pending_buffer(
            args.output_dir,
            split_name,
            buffers[split_name],
            compress=args.compress,
        )

    manifest = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "max_games": args.max_games,
        "workers": args.workers,
        "chunk_games": args.chunk_games,
        "games_per_shard": args.games_per_shard,
        "processed_games": processed_games,
        "feature_count": len(SEQUENCE_FEATURES),
        "splits": {
            split: {
                "games": split_game_counts[split],
                "shards": shard_indices[split],
            }
            for split in ["train", "val", "test"]
        },
    }
    maybe_save_checkpoint(force=True)
    maybe_report_progress(force=True)
    (args.output_dir / "manifest.json").write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))
    if checkpoint_path(args.output_dir).exists():
        checkpoint_path(args.output_dir).unlink()
    print(
        f"Finished. Processed games: {processed_games} | "
        f"Saved train/val/test games: "
        f"{split_game_counts['train']}/{split_game_counts['val']}/{split_game_counts['test']} | "
        f"Completed train/val/test shards: "
        f"{shard_indices['train']}/{shard_indices['val']}/{shard_indices['test']}",
        flush=True,
    )
    print(orjson.dumps(manifest, option=orjson.OPT_INDENT_2).decode(), flush=True)


if __name__ == "__main__":
    main()
