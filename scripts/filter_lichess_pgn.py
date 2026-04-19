"""Filter a large Lichess PGN file using streaming header checks.

This script is designed for very large PGN exports that do not fit in memory.
It reads one game at a time, inspects the PGN headers, and writes only the
games that match the requested criteria.

Example:
    python scripts/filter_lichess_pgn.py ^
        --input lichess_db_standard_rated_2026-02/lichess_db_standard_rated_2026-02.pgn ^
        --output data/lichess_rapid_10_0_completed.pgn ^
        --time-control 600+0 ^
        --termination Normal
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the PGN filtering script."""
    parser = argparse.ArgumentParser(
        description="Stream-filter a PGN file by time control and termination."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the source PGN file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path where the filtered PGN file will be written.",
    )
    parser.add_argument(
        "--time-control",
        default="600+0",
        help='PGN TimeControl value to keep, for example "600+0".',
    )
    parser.add_argument(
        "--termination",
        action="append",
        dest="terminations",
        default=None,
        help=(
            "Termination value to keep. Repeat this flag to allow multiple "
            'values. Defaults to only "Normal".'
        ),
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Optional cap on the number of kept games written to the output.",
    )
    return parser.parse_args()


def iter_games(pgn_path: Path):
    """Yield PGN games as lists of lines from a large PGN file.

    Args:
        pgn_path: Path to the input PGN file.

    Yields:
        list[str]: The raw lines corresponding to a single PGN game.

    Example:
        >>> isinstance(next(iter_games(Path("sample.pgn"))), list)
        True
    """
    current_game: list[str] = []
    seen_moves = False

    with pgn_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("[Event ") and current_game and seen_moves:
                yield current_game
                current_game = []
                seen_moves = False

            current_game.append(line)

            if line.strip() == "" and current_game and any(
                game_line.startswith("[") for game_line in current_game
            ):
                continue

            if not line.startswith("[") and line.strip():
                seen_moves = True

        if current_game:
            yield current_game


def extract_headers(game_lines: list[str]) -> dict[str, str]:
    """Extract PGN headers from a game block.

    Args:
        game_lines: Raw PGN lines for one game.

    Returns:
        dict[str, str]: Mapping from PGN tag names to values.

    Example:
        >>> extract_headers(['[Event "Rated Rapid game"]\\n'])["Event"]
        'Rated Rapid game'
    """
    headers: dict[str, str] = {}

    for line in game_lines:
        stripped = line.strip()
        if not stripped.startswith("[") or not stripped.endswith("]"):
            continue

        try:
            key, value = stripped[1:-1].split(" ", 1)
        except ValueError:
            continue

        headers[key] = value.strip().strip('"')

    return headers


def should_keep_game(
    headers: dict[str, str],
    expected_time_control: str,
    allowed_terminations: set[str],
) -> bool:
    """Return whether a game matches the requested filters.

    Args:
        headers: PGN header mapping for one game.
        expected_time_control: Required value of the TimeControl PGN tag.
        allowed_terminations: Allowed values for the Termination PGN tag.

    Returns:
        bool: True when the game matches the filters.

    Example:
        >>> should_keep_game(
        ...     {"TimeControl": "600+0", "Termination": "Normal"},
        ...     "600+0",
        ...     {"Normal"},
        ... )
        True
    """
    return (
        headers.get("TimeControl") == expected_time_control
        and headers.get("Termination") in allowed_terminations
    )


def main() -> None:
    """Filter the input PGN file and write matching games to disk."""
    args = parse_args()
    allowed_terminations = set(args.terminations or ["Normal"])

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_games = 0
    kept_games = 0

    with args.output.open("w", encoding="utf-8", newline="") as out_handle:
        for game_lines in iter_games(args.input):
            total_games += 1
            headers = extract_headers(game_lines)

            if not should_keep_game(
                headers=headers,
                expected_time_control=args.time_control,
                allowed_terminations=allowed_terminations,
            ):
                continue

            out_handle.writelines(game_lines)
            if game_lines and game_lines[-1].strip():
                out_handle.write("\n")
            kept_games += 1

            if args.max_games is not None and kept_games >= args.max_games:
                break

    print(f"Processed games: {total_games}")
    print(f"Kept games: {kept_games}")
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
