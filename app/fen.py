from __future__ import annotations

PIECE_TO_FEN = {
    "white_pawn": "P", "white_knight": "N", "white_bishop": "B",
    "white_rook": "R", "white_queen": "Q", "white_king": "K",
    "black_pawn": "p", "black_knight": "n", "black_bishop": "b",
    "black_rook": "r", "black_queen": "q", "black_king": "k", "empty": "",
}


def validate_board_matrix(board_matrix: list[list[str]]) -> None:
    if len(board_matrix) != 8 or any(len(row) != 8 for row in board_matrix):
        raise ValueError("Board matrix must be exactly 8x8.")
    unknown = {p for row in board_matrix for p in row if p not in PIECE_TO_FEN}
    if unknown:
        raise ValueError(f"Unknown piece labels: {sorted(unknown)}")


def matrix_to_fen(board_matrix, *, turn="w", castling="-", en_passant="-", halfmove=0, fullmove=1):
    validate_board_matrix(board_matrix)
    if turn not in {"w", "b"}:
        raise ValueError("turn must be 'w' or 'b'.")
    if halfmove < 0 or fullmove < 1:
        raise ValueError("Invalid FEN move counters.")

    ranks = []
    for row in board_matrix:
        empty = 0
        rank = []
        for piece in row:
            if piece == "empty":
                empty += 1
            else:
                if empty:
                    rank.append(str(empty))
                    empty = 0
                rank.append(PIECE_TO_FEN[piece])
        if empty:
            rank.append(str(empty))
        ranks.append("".join(rank))
    return f"{'/'.join(ranks)} {turn} {castling} {en_passant} {halfmove} {fullmove}"
