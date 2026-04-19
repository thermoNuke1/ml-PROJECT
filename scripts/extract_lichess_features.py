"""Extract streaming baseline features from a filtered Lichess PGN file.

This script creates a tabular dataset of move-level snapshots from a PGN file.
It is designed to work without loading the entire file into memory and without
requiring external chess-specific libraries.

Each output row represents the game state immediately after a move has been
played, using only information available at that moment. The initial version
focuses on metadata and clock-based features, which are already strong signals
for a human-centered win prediction baseline.

Example:
    python scripts/extract_lichess_features.py ^
        --input data/sample_rapid_10_0_completed.pgn ^
        --output data/sample_move_features.csv ^
        --max-games 100
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


CLOCK_PATTERN = re.compile(r"\[%clk\s+(\d+):(\d+):(\d+)\]")
MOVE_NUMBER_PATTERN = re.compile(r"^\d+\.+$")
RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}
HEADER_PATTERN = re.compile(r'^\[(\w+)\s+"(.*)"\]$')

OUTPUT_COLUMNS = [
    "game_id",
    "date",
    "white_player",
    "black_player",
    "white_elo",
    "black_elo",
    "elo_diff_white_minus_black",
    "result",
    "white_win",
    "black_win",
    "draw",
    "time_control",
    "termination",
    "ply_index",
    "fullmove_number",
    "mover",
    "side_to_move",
    "san",
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
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for feature extraction.

    Returns:
        argparse.Namespace: Parsed arguments.

    Example:
        >>> isinstance(parse_args, object)
        True
    """
    parser = argparse.ArgumentParser(
        description="Extract move-level baseline features from a PGN file."
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
        help="Destination CSV file for extracted features.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Optional cap on the number of games processed.",
    )
    return parser.parse_args()


def parse_time_control_seconds(time_control: str) -> int | None:
    """Parse the initial base time from a Lichess TimeControl tag.

    Args:
        time_control: PGN TimeControl value such as "600+0".

    Returns:
        int | None: Initial clock in seconds, or None if parsing fails.

    Example:
        >>> parse_time_control_seconds("600+0")
        600
    """
    if "+" not in time_control:
        return None

    base, _increment = time_control.split("+", 1)
    try:
        return int(base)
    except ValueError:
        return None


def parse_clock_seconds(comment: str) -> int | None:
    """Extract a clock value in seconds from a PGN comment.

    Args:
        comment: PGN comment text containing a [%clk ...] annotation.

    Returns:
        int | None: Clock in seconds if present, else None.

    Example:
        >>> parse_clock_seconds('{ [%clk 0:09:58] }')
        598
    """
    match = CLOCK_PATTERN.search(comment)
    if not match:
        return None

    hours, minutes, seconds = (int(value) for value in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def parse_result_flags(result: str) -> tuple[int, int, int]:
    """Convert a PGN result into binary target flags.

    Args:
        result: PGN Result value.

    Returns:
        tuple[int, int, int]: White win, black win, draw flags.

    Example:
        >>> parse_result_flags("1-0")
        (1, 0, 0)
    """
    if result == "1-0":
        return 1, 0, 0
    if result == "0-1":
        return 0, 1, 0
    if result == "1/2-1/2":
        return 0, 0, 1
    return 0, 0, 0


def iter_games(pgn_path: Path):
    """Yield PGN games as raw text blocks.

    Args:
        pgn_path: Path to a PGN file.

    Yields:
        list[str]: Lines associated with a single PGN game.

    Example:
        >>> callable(iter_games)
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

            if not line.startswith("[") and line.strip():
                seen_moves = True

        if current_game:
            yield current_game


def split_game(game_lines: list[str]) -> tuple[dict[str, str], str]:
    """Split a PGN game into headers and move text.

    Args:
        game_lines: Raw PGN lines for one game.

    Returns:
        tuple[dict[str, str], str]: Parsed headers and normalized move text.

    Example:
        >>> headers, moves = split_game(['[Result "1-0"]\\n', '\\n', '1. e4 1-0\\n'])
        >>> headers["Result"], moves
        ('1-0', '1. e4 1-0')
    """
    headers: dict[str, str] = {}
    move_lines: list[str] = []
    in_moves = False

    for raw_line in game_lines:
        line = raw_line.rstrip("\n")
        if not in_moves:
            match = HEADER_PATTERN.match(line.strip())
            if match:
                key, value = match.groups()
                headers[key] = value
                continue

            if line.strip() == "":
                if headers:
                    in_moves = True
                continue

        if in_moves and line.strip():
            move_lines.append(line.strip())

    return headers, " ".join(move_lines)


def tokenize_moves(move_text: str) -> list[str]:
    """Tokenize PGN move text while preserving comments.

    Args:
        move_text: PGN move text section for one game.

    Returns:
        list[str]: Tokens including SAN moves, comments, and result markers.

    Example:
        >>> tokenize_moves('1. e4 { [%clk 0:10:00] } e5 1-0')[:4]
        ['1.', 'e4', '{ [%clk 0:10:00] }', 'e5']
    """
    tokens: list[str] = []
    current: list[str] = []
    in_comment = False

    for char in move_text:
        if char == "{":
            if current:
                tokens.append("".join(current).strip())
                current = []
            in_comment = True
            current.append(char)
            continue

        if char == "}":
            current.append(char)
            tokens.append("".join(current).strip())
            current = []
            in_comment = False
            continue

        if in_comment:
            current.append(char)
            continue

        if char.isspace():
            if current:
                tokens.append("".join(current).strip())
                current = []
            continue

        current.append(char)

    if current:
        tokens.append("".join(current).strip())

    return [token for token in tokens if token]


def safe_int(value: str | None) -> int | None:
    """Convert a string to int when possible.

    Args:
        value: Input value to convert.

    Returns:
        int | None: Converted integer or None.

    Example:
        >>> safe_int("1800")
        1800
    """
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def extract_san_features(san: str) -> dict[str, int]:
    """Extract simple move-shape features from SAN notation.

    Args:
        san: Standard algebraic notation for a move.

    Returns:
        dict[str, int]: Simple binary and length-based SAN features.

    Example:
        >>> extract_san_features("Qxe5+")["is_capture"]
        1
    """
    normalized = san.strip()
    return {
        "is_capture": int("x" in normalized),
        "is_check": int("+" in normalized),
        "is_checkmate": int("#" in normalized),
        "is_castle": int(normalized.startswith("O-O")),
        "is_promotion": int("=" in normalized),
        "san_length": len(normalized),
    }


def build_rows_for_game(headers: dict[str, str], move_text: str) -> list[dict[str, object]]:
    """Build move-level feature rows for one game.

    Args:
        headers: PGN header mapping.
        move_text: PGN move text.

    Returns:
        list[dict[str, object]]: Extracted feature rows.

    Example:
        >>> rows = build_rows_for_game(
        ...     {
        ...         "Site": "https://lichess.org/test",
        ...         "Date": "2026.02.01",
        ...         "White": "Alice",
        ...         "Black": "Bob",
        ...         "WhiteElo": "1800",
        ...         "BlackElo": "1750",
        ...         "Result": "1-0",
        ...         "TimeControl": "600+0",
        ...         "Termination": "Normal",
        ...     },
        ...     '1. e4 { [%clk 0:09:58] } e5 { [%clk 0:09:57] } 1-0',
        ... )
        >>> len(rows)
        2
    """
    tokens = tokenize_moves(move_text)
    initial_time = parse_time_control_seconds(headers.get("TimeControl", ""))
    white_time_seconds = initial_time
    black_time_seconds = initial_time
    white_elo = safe_int(headers.get("WhiteElo"))
    black_elo = safe_int(headers.get("BlackElo"))
    result = headers.get("Result", "*")
    white_win, black_win, draw = parse_result_flags(result)
    rows: list[dict[str, object]] = []
    pending_row: dict[str, object] | None = None
    ply_index = 0

    for token in tokens:
        if token.startswith("{") and token.endswith("}"):
            if pending_row is None:
                continue

            clock_value = parse_clock_seconds(token)
            if clock_value is None:
                continue

            previous_mover_time = (
                white_time_seconds
                if pending_row["mover"] == "white"
                else black_time_seconds
            )

            if pending_row["mover"] == "white":
                white_time_seconds = clock_value
            else:
                black_time_seconds = clock_value

            row = pending_row.copy()
            row["white_time_seconds"] = white_time_seconds
            row["black_time_seconds"] = black_time_seconds
            row["mover_time_seconds"] = (
                white_time_seconds if row["mover"] == "white" else black_time_seconds
            )
            row["opponent_time_seconds"] = (
                black_time_seconds if row["mover"] == "white" else white_time_seconds
            )
            row["mover_time_spent_seconds"] = (
                None
                if previous_mover_time is None
                else previous_mover_time - row["mover_time_seconds"]
            )
            row["white_time_ratio"] = (
                None
                if initial_time in (None, 0)
                else white_time_seconds / initial_time
            )
            row["black_time_ratio"] = (
                None
                if initial_time in (None, 0)
                else black_time_seconds / initial_time
            )
            row["clock_diff_seconds_white_minus_black"] = (
                None
                if white_time_seconds is None or black_time_seconds is None
                else white_time_seconds - black_time_seconds
            )
            rows.append(row)
            pending_row = None
            continue

        if MOVE_NUMBER_PATTERN.match(token):
            continue

        if token in RESULT_TOKENS:
            continue

        ply_index += 1
        mover = "white" if ply_index % 2 == 1 else "black"
        side_to_move = "black" if mover == "white" else "white"
        fullmove_number = (ply_index + 1) // 2

        pending_row = {
            "game_id": headers.get("Site", ""),
            "date": headers.get("Date", ""),
            "white_player": headers.get("White", ""),
            "black_player": headers.get("Black", ""),
            "white_elo": white_elo,
            "black_elo": black_elo,
            "elo_diff_white_minus_black": (
                None
                if white_elo is None or black_elo is None
                else white_elo - black_elo
            ),
            "result": result,
            "white_win": white_win,
            "black_win": black_win,
            "draw": draw,
            "time_control": headers.get("TimeControl", ""),
            "termination": headers.get("Termination", ""),
            "ply_index": ply_index,
            "fullmove_number": fullmove_number,
            "mover": mover,
            "side_to_move": side_to_move,
            "san": token,
            **extract_san_features(token),
        }

    return rows


def main() -> None:
    """Run streaming feature extraction and write a CSV file.

    Example:
        >>> callable(main)
        True
    """
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    processed_games = 0
    written_rows = 0

    with args.output.open("w", encoding="utf-8", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for game_lines in iter_games(args.input):
            headers, move_text = split_game(game_lines)
            rows = build_rows_for_game(headers, move_text)

            for row in rows:
                writer.writerow(row)
                written_rows += 1

            processed_games += 1
            if args.max_games is not None and processed_games >= args.max_games:
                break

    print(f"Processed games: {processed_games}")
    print(f"Written rows: {written_rows}")
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
