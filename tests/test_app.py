import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app
# ... rest of your code
import pytest
from fastapi.testclient import TestClient
from app.fen import matrix_to_fen
from app.chess_engine import analyze_position

client = TestClient(app)

class TestAPI:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_predict_no_file(self):
        response = client.post("/predict")
        assert response.status_code == 422  # Unprocessable entity

class TestFEN:
    def test_starting_position(self):
        """Test FEN generation for chess starting position."""
        test_matrix = [
            ["black_rook", "black_knight", "black_bishop", "black_queen", "black_king", "black_bishop", "black_knight", "black_rook"],
            ["black_pawn", "black_pawn", "black_pawn", "black_pawn", "black_pawn", "black_pawn", "black_pawn", "black_pawn"],
            ["empty"] * 8,
            ["empty"] * 8,
            ["empty"] * 8,
            ["empty"] * 8,
            ["white_pawn"] * 8,
            ["white_rook", "white_knight", "white_bishop", "white_queen", "white_king", "white_bishop", "white_knight", "white_rook"]
        ]
        
        fen = matrix_to_fen(test_matrix)
        assert fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"
    
    def test_empty_board(self):
        """Test FEN generation for empty board."""
        test_matrix = [["empty"] * 8 for _ in range(8)]
        fen = matrix_to_fen(test_matrix)
        assert fen == "8/8/8/8/8/8/8/8 w - - 0 1"

class TestChessEngine:
    def test_valid_position(self):
        """Test chess engine analysis on starting position."""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"
        analysis = analyze_position(fen)
        
        assert analysis["is_valid"] == True
        assert analysis["turn"] == "White"
        assert analysis["in_check"] == False
        assert analysis["is_checkmate"] == False
        assert analysis["legal_moves_count"] == 20
    
    def test_invalid_fen(self):
        """Test error handling for invalid FEN."""
        fen = "invalid fen string"
        analysis = analyze_position(fen)
        assert "error" in analysis

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-W", "ignore::pytest.PytestAssertRewriteWarning"])
