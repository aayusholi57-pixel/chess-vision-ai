from __future__ import annotations
import chess


def analyze_position(fen_string: str) -> dict:
    try:
        board = chess.Board(fen_string)
    except ValueError as exc:
        return {"is_valid": False, "error": str(exc)}

    legal_moves = [move.uci() for move in board.legal_moves]
    return {
        "is_valid": board.is_valid(),
        "turn": "White" if board.turn == chess.WHITE else "Black",
        "in_check": board.is_check(),
        "is_checkmate": board.is_checkmate(),
        "is_stalemate": board.is_stalemate(),
        "is_insufficient_material": board.is_insufficient_material(),
        "legal_moves_count": len(legal_moves),
        "sample_legal_moves": legal_moves[:10],
    }
