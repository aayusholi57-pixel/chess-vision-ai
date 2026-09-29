def matrix_to_fen(board_matrix):
    """
    Converts an 8x8 matrix of piece strings into a standard FEN string.
    """
    # 1. Dictionary to translate our class names into official FEN characters
    piece_to_fen = {
        "white_pawn": "P", "white_knight": "N", "white_bishop": "B", 
        "white_rook": "R", "white_queen": "Q", "white_king": "K",
        "black_pawn": "p", "black_knight": "n", "black_bishop": "b", 
        "black_rook": "r", "black_queen": "q", "black_king": "k",
        "empty": ""
    }
    
    fen_rows = []
    
    # 2. Loop through each row in our 8x8 matrix
    for row in board_matrix:
        empty_count = 0
        fen_row = ""
        
        for square in row:
            if square == "empty":
                empty_count += 1
            else:
                # If we have empty squares counted up, append the number first
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                
                # Append the piece character
                fen_row += piece_to_fen[square]
                
        # If a row ends with empty squares, append the final count
        if empty_count > 0:
            fen_row += str(empty_count)
            
        fen_rows.append(fen_row)
        
    # 3. Join all 8 rows together with a forward slash "/"
    fen_position = "/".join(fen_rows)
    
    # 4. Add the default game state (White to move, no castling rights, etc.)
    # (Since we only have an image, we don't know whose turn it is, so we default to White 'w')
    full_fen = f"{fen_position} w - - 0 1"
    
    return full_fen

if __name__ == "__main__":
    # Let's test it with a fake starting board
    test_matrix = [
        ["black_rook", "black_knight", "black_bishop", "black_queen", "black_king", "black_bishop", "black_knight", "black_rook"],
        ["black_pawn", "black_pawn", "black_pawn", "black_pawn", "black_pawn", "black_pawn", "black_pawn", "black_pawn"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty"],
        ["white_pawn", "white_pawn", "white_pawn", "white_pawn", "white_pawn", "white_pawn", "white_pawn", "white_pawn"],
        ["white_rook", "white_knight", "white_bishop", "white_queen", "white_king", "white_bishop", "white_knight", "white_rook"]
    ]
    
    print("Testing FEN Generation...")
    fen = matrix_to_fen(test_matrix)
    print(f"Generated FEN: {fen}")
    print("Expected FEN:  rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1")