"""Parallel board-aware PGN feature extraction using multiprocessing.

This script keeps the same feature definition as `extract_lichess_board_features.py`
but parallelizes the expensive python-chess board reconstruction step across
multiple worker processes. It is intended as a faster drop-in extractor for
larger board-aware experiments.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import chess.pgn

from extract_lichess_board_features import (
    OUTPUT_COLUMNS,
    build_row,
    parse_time_control_seconds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract board-aware move features in parallel."
    )
    parser.add_argument("--input", required=True, type=Path, help="Source PGN file.")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination CSV file for extracted features.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Optional cap on games processed.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of worker processes to use.",
    )
    parser.add_argument(
        "--chunk-games",
        type=int,
        default=100,
        help="Number of games per worker task.",
    )
    return parser.parse_args()


def game_to_text(game: chess.pgn.Game) -> str:
    exporter = chess.pgn.StringExporter(
        headers=True,
        variations=False,
        comments=True,
    )
    return game.accept(exporter)


def process_game_text(game_text: str) -> list[dict[str, object]]:
    handle = io.StringIO(game_text)
    game = chess.pgn.read_game(handle)
    if game is None:
        return []

    headers = game.headers
    initial_time = parse_time_control_seconds(headers.get("TimeControl", ""))
    white_time_seconds = initial_time
    black_time_seconds = initial_time
    previous_white_time = initial_time
    previous_black_time = initial_time

    rows: list[dict[str, object]] = []
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

        rows.append(
            build_row(
                node=node,
                headers=headers,
                white_time_seconds=white_time_seconds,
                black_time_seconds=black_time_seconds,
                previous_white_time=previous_white_time,
                previous_black_time=previous_black_time,
                initial_time=initial_time,
            )
        )
    return rows


def process_chunk(game_texts: list[str]) -> list[dict[str, object]]:
    chunk_rows: list[dict[str, object]] = []
    for game_text in game_texts:
        chunk_rows.extend(process_game_text(game_text))
    return chunk_rows


def yield_game_chunks(
    input_path: Path,
    max_games: int | None,
    chunk_games: int,
):
    processed_games = 0
    current_chunk: list[str] = []

    with input_path.open("r", encoding="utf-8", errors="replace") as pgn_handle:
        while True:
            game = chess.pgn.read_game(pgn_handle)
            if game is None:
                break

            current_chunk.append(game_to_text(game))
            processed_games += 1

            if len(current_chunk) >= chunk_games:
                yield current_chunk, processed_games
                current_chunk = []

            if max_games is not None and processed_games >= max_games:
                break

    if current_chunk:
        yield current_chunk, processed_games


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written_rows = 0
    submitted_games = 0

    with args.output.open("w", encoding="utf-8", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for game_chunk, processed_games in yield_game_chunks(
                input_path=args.input,
                max_games=args.max_games,
                chunk_games=args.chunk_games,
            ):
                submitted_games = processed_games
                futures.append(executor.submit(process_chunk, game_chunk))

            for future in futures:
                rows = future.result()
                for row in rows:
                    writer.writerow(row)
                written_rows += len(rows)

    print(f"Processed games: {submitted_games}")
    print(f"Written rows: {written_rows}")
    print(f"Workers used: {args.workers}")
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
