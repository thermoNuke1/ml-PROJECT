"""Extract board-aware move features from a filtered Lichess PGN into a CSV.

Replays each game with python-chess and writes one row per move with Elo,
clock, SAN flags, and board state (material, castling rights, legal moves, etc).
Used for snapshot-style models (logistic regression, SVM, MLP, XGBoost).

    python scripts/extract_lichess_board_features.py \
        --input data/lichess_rapid_10_0_completed.pgn \
        --output data/dev_board_features_10000_games.csv \
        --max-games 10000
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import chess
import chess.pgn
import io
import multiprocessing



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
    "uci",
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
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for board-aware feature extraction."""
    parser = argparse.ArgumentParser(
        description="Extract move-level board features from a PGN file."
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
        help="Optional cap on the number of games processed.",
    )
    # convenient for testing, leave blank to use all CPU cores for multiprocessing
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: all CPU cores).",
    )
    return parser.parse_args()


def safe_int(value: str | None) -> int | None:
    """Convert a string to int when possible."""
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_time_control_seconds(time_control: str) -> int | None:
    """Parse the base time in seconds from a PGN time control tag."""
    if "+" not in time_control:
        return None
    base, _increment = time_control.split("+", 1)
    try:
        return int(base)
    except ValueError:
        return None


def parse_result_flags(result: str) -> tuple[int, int, int]:
    """Convert a PGN result string into binary target flags."""
    if result == "1-0":
        return 1, 0, 0
    if result == "0-1":
        return 0, 1, 0
    if result == "1/2-1/2":
        return 0, 0, 1
    return 0, 0, 0



def process_game(game: chess.pgn.Game) -> list[dict]:
    """Process a single game into move-level rows using incremental board state."""
    headers = game.headers
    initial_time = parse_time_control_seconds(headers.get("TimeControl", ""))
    white_time_seconds = initial_time
    black_time_seconds = initial_time
    previous_white_time = initial_time
    previous_black_time = initial_time

    result = headers.get("Result", "*")
    white_win, black_win, draw = parse_result_flags(result)
    white_elo = safe_int(headers.get("WhiteElo"))
    black_elo = safe_int(headers.get("BlackElo"))
    elo_diff = None if white_elo is None or black_elo is None else white_elo - black_elo

    # Read header strings once per game instead of once per move
    game_id = headers.get("Site", "")
    date = headers.get("Date", "")
    white_player = headers.get("White", "")
    black_player = headers.get("Black", "")
    time_control = headers.get("TimeControl", "")
    termination = headers.get("Termination", "")

    rows: list[dict] = []
    # Maintain one running board — push() is O(1) vs node.board() which is O(n)
    board = game.board()
    ply_index = 0 # faster than node.ply() which is O(n)

    for node in game.mainline():
        move = node.move

        # Capture parent-board state BEFORE pushing the move
        mover_is_white = board.turn == chess.WHITE
        mover = "white" if mover_is_white else "black"
        is_capture = int(board.is_capture(move))
        is_castle = int(board.is_castling(move))

        clock_value = node.clock()
        if clock_value is not None:
            if mover_is_white:
                previous_white_time = white_time_seconds
                white_time_seconds = int(clock_value)
            else:
                previous_black_time = black_time_seconds
                black_time_seconds = int(clock_value)

        board.push(move)  # advance board; all queries below use post-move state
        ply_index += 1
        san = node.san()

        in_check = board.is_check()
        side_to_move = "white" if board.turn == chess.WHITE else "black"
        mover_time_seconds = white_time_seconds if mover_is_white else black_time_seconds
        opponent_time_seconds = black_time_seconds if mover_is_white else white_time_seconds
        previous_mover_time = previous_white_time if mover_is_white else previous_black_time

        # Inline piece counts to avoid allocating a temporary dict and unpacking
        # it with **. Each board.pieces() call is already C-level via the extension.
        white_pawns = len(board.pieces(chess.PAWN, chess.WHITE))
        black_pawns = len(board.pieces(chess.PAWN, chess.BLACK))
        white_knights = len(board.pieces(chess.KNIGHT, chess.WHITE))
        black_knights = len(board.pieces(chess.KNIGHT, chess.BLACK))
        white_bishops = len(board.pieces(chess.BISHOP, chess.WHITE))
        black_bishops = len(board.pieces(chess.BISHOP, chess.BLACK))
        white_rooks = len(board.pieces(chess.ROOK, chess.WHITE))
        black_rooks = len(board.pieces(chess.ROOK, chess.BLACK))
        white_queens = len(board.pieces(chess.QUEEN, chess.WHITE))
        black_queens = len(board.pieces(chess.QUEEN, chess.BLACK))
        white_material = white_pawns + white_knights*3 + white_bishops*3 + white_rooks*5 + white_queens*9
        black_material = black_pawns + black_knights*3 + black_bishops*3 + black_rooks*5 + black_queens*9

        row: dict = {
            "game_id": game_id,
            "date": date,
            "white_player": white_player,
            "black_player": black_player,
            "white_elo": white_elo,
            "black_elo": black_elo,
            "elo_diff_white_minus_black": elo_diff,
            "result": result,
            "white_win": white_win,
            "black_win": black_win,
            "draw": draw,
            "time_control": time_control,
            "termination": termination,
            "ply_index": ply_index,
            "fullmove_number": board.fullmove_number,
            "mover": mover,
            "side_to_move": side_to_move,
            "san": san,
            "uci": move.uci(),
            "is_capture": is_capture,
            "is_check": int(in_check),
            "is_checkmate": int(in_check and board.is_checkmate()),
            "is_castle": is_castle,
            "is_promotion": int(move.promotion is not None),
            "san_length": len(san),
            "white_time_seconds": white_time_seconds,
            "black_time_seconds": black_time_seconds,
            "mover_time_seconds": mover_time_seconds,
            "opponent_time_seconds": opponent_time_seconds,
            "mover_time_spent_seconds": (
                None
                if previous_mover_time is None or mover_time_seconds is None
                else previous_mover_time - mover_time_seconds
            ),
            "white_time_ratio": (
                None
                if initial_time in (None, 0) or white_time_seconds is None
                else white_time_seconds / initial_time
            ),
            "black_time_ratio": (
                None
                if initial_time in (None, 0) or black_time_seconds is None
                else black_time_seconds / initial_time
            ),
            "clock_diff_seconds_white_minus_black": (
                None
                if white_time_seconds is None or black_time_seconds is None
                else white_time_seconds - black_time_seconds
            ),
            "legal_moves_count": board.legal_moves.count(),
            "halfmove_clock": board.halfmove_clock,
            "white_material": white_material,
            "black_material": black_material,
            "material_diff_white_minus_black": white_material - black_material,
            "white_pawns": white_pawns,
            "black_pawns": black_pawns,
            "white_knights": white_knights,
            "black_knights": black_knights,
            "white_bishops": white_bishops,
            "black_bishops": black_bishops,
            "white_rooks": white_rooks,
            "black_rooks": black_rooks,
            "white_queens": white_queens,
            "black_queens": black_queens,
            "white_has_bishop_pair": int(white_bishops >= 2),
            "black_has_bishop_pair": int(black_bishops >= 2),
            "white_can_castle_kingside": int(board.has_kingside_castling_rights(chess.WHITE)),
            "white_can_castle_queenside": int(board.has_queenside_castling_rights(chess.WHITE)),
            "black_can_castle_kingside": int(board.has_kingside_castling_rights(chess.BLACK)),
            "black_can_castle_queenside": int(board.has_queenside_castling_rights(chess.BLACK)),
            "is_insufficient_material": int(board.is_insufficient_material()),
            "mover_is_white": int(mover_is_white),
            "side_to_move_is_white": int(board.turn == chess.WHITE),
        }
        rows.append(row)

    return rows

def _process_game_string(game_str: str) -> list[dict]:
    """Worker entry point: parse a PGN string and extract features."""
    pgn_io = io.StringIO(game_str)
    game = chess.pgn.read_game(pgn_io)
    if game is None:
        return []
    return process_game(game)


def _game_strings_iter(pgn_path: Path, max_games: int | None):
    """Yield each game as a PGN string, reading the source file sequentially."""
    # StringExporterMixin has no begin_game() reset — self.lines accumulates
    # across calls if the same instance is reused. Create one per game.
    with pgn_path.open("r", encoding="utf-8", errors="replace") as f:
        count = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            yield game.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=True))
            count += 1
            if max_games is not None and count >= max_games:
                break

def main() -> None:
    """Extract move-level board features from PGN and write them to CSV."""
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    num_workers = args.workers or multiprocessing.cpu_count()
    processed_games = 0
    written_rows = 0

    with args.output.open("w", encoding="utf-8", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        game_iter = _game_strings_iter(args.input, args.max_games)
        if num_workers == 1:
            for game_str in game_iter:
                rows = _process_game_string(game_str)
                if rows:
                    writer.writerows(rows)
                    processed_games += 1
                    written_rows += len(rows)
        else:
            with multiprocessing.Pool(num_workers) as pool:
                for rows in pool.imap(_process_game_string, game_iter, chunksize=16):
                    if rows:
                        writer.writerows(rows)
                        processed_games += 1
                        written_rows += len(rows)
    
    print(f"Workers used: {num_workers}")
    print(f"Processed games: {processed_games}")
    print(f"Written rows: {written_rows}")
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
