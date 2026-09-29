import pytest
from fastapi.testclient import TestClient

from app.board import load_chess_model
from app.chess_engine import analyze_position
from app.fen import matrix_to_fen
from app.main import app
from app.preprocessing import process_uploaded_image

client = TestClient(app)


def starting_matrix():
    return [
        ["black_rook", "black_knight", "black_bishop", "black_queen", "black_king", "black_bishop", "black_knight", "black_rook"],
        ["black_pawn"] * 8,
        ["empty"] * 8,
        ["empty"] * 8,
        ["empty"] * 8,
        ["empty"] * 8,
        ["white_pawn"] * 8,
        ["white_rook", "white_knight", "white_bishop", "white_queen", "white_king", "white_bishop", "white_knight", "white_rook"],
    ]


def test_starting_position_fen():
    assert matrix_to_fen(starting_matrix()) == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"


def test_empty_board_fen():
    assert matrix_to_fen([["empty"] * 8 for _ in range(8)]) == "8/8/8/8/8/8/8/8 w - - 0 1"


def test_invalid_matrix():
    with pytest.raises(ValueError):
        matrix_to_fen([["empty"] * 7 for _ in range(8)])


def test_invalid_piece():
    board = [["empty"] * 8 for _ in range(8)]
    board[0][0] = "not_a_piece"
    with pytest.raises(ValueError):
        matrix_to_fen(board)


def test_chess_analysis():
    result = analyze_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1")
    assert result["is_valid"] is True
    assert result["legal_moves_count"] == 20


def test_invalid_fen():
    result = analyze_position("not a fen")
    assert result["is_valid"] is False
    assert "error" in result


def test_model_missing_is_safe():
    _, _, status = load_chess_model(force_reload=True)
    assert isinstance(status, dict)
    assert "ready" in status


def test_preprocessing_rejects_non_image(tmp_path):
    bad = tmp_path / "not-an-image.jpg"
    bad.write_text("not an image")
    with pytest.raises(ValueError):
        process_uploaded_image(bad, tmp_path / "squares")


def test_health_without_model_is_not_healthy():
    assert client.get("/health").status_code == 503


def test_predict_without_file():
    assert client.post("/predict").status_code == 422


def test_predict_success_path(monkeypatch):
    board = starting_matrix()
    confidence = [[0.99] * 8 for _ in range(8)]
    monkeypatch.setattr("app.main.process_uploaded_image", lambda *_args, **_kwargs: "fake-squares")
    monkeypatch.setattr("app.main.predict_squares", lambda *_args, **_kwargs: (board, confidence))
    response = client.post(
        "/predict",
        files={"file": ("board.jpg", b"fake-image", "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["fen"].startswith("rnbqkbnr/")


def test_predict_rejects_non_image():
    response = client.post(
        "/predict",
        files={"file": ("board.txt", b"not-an-image", "text/plain")},
    )
    assert response.status_code == 415


def test_predict_rejects_large_upload(monkeypatch):
    monkeypatch.setattr("app.main.MAX_UPLOAD_BYTES", 1)
    response = client.post(
        "/predict",
        files={"file": ("board.jpg", b"12", "image/jpeg")},
    )
    assert response.status_code == 413


def test_batched_square_prediction(monkeypatch, tmp_path):
    import torch
    from PIL import Image
    import app.board as board_module

    for row in range(8):
        for col in range(8):
            Image.new("RGB", (100, 100), (128, 128, 128)).save(
                tmp_path / f"square_r{row}_c{col}.jpg"
            )

    class FakeModel:
        def __call__(self, batch):
            return torch.zeros((batch.shape[0], 13))

    monkeypatch.setattr(
        board_module,
        "load_chess_model",
        lambda: (FakeModel(), board_module.CLASSES, {"ready": True}),
    )
    board, confidence = board_module.predict_squares(tmp_path)
    assert len(board) == 8
    assert all(len(row) == 8 for row in board)
    assert len(confidence) == 8
    assert all(len(row) == 8 for row in confidence)


def test_preprocessing_extracts_64_squares(tmp_path):
    import cv2
    import numpy as np

    source = tmp_path / "board.jpg"
    image = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (100, 100), (900, 900), (0, 0, 0), 12)
    cv2.imwrite(str(source), image)

    output = process_uploaded_image(source, tmp_path / "squares")
    assert len(list(tmp_path.joinpath("squares").glob("square_r*_c*.jpg"))) == 64
    assert output.endswith("squares")
