from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.board import load_chess_model, predict_squares
from app.chess_engine import analyze_position
from app.config import ALLOWED_ORIGINS, MAX_UPLOAD_BYTES, MIN_CONFIDENCE
from app.fen import matrix_to_fen
from app.preprocessing import process_uploaded_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chess Vision AI",
    version="2.0.0",
    description="Chessboard image recognition, FEN reconstruction and rule validation.",
)

if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


@app.get("/health")
def health():
    _, _, status = load_chess_model()
    if not status or not status.get("ready"):
        raise HTTPException(status_code=503, detail=status or {"ready": False})
    return {"status": "healthy", "model": status}


@app.get("/ready")
def ready():
    return health()


@app.post("/predict")
async def predict_board(file: UploadFile = File(...)):
    content_type = (file.content_type or "").lower()
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image uploads are supported.")

    temp_dir = Path(tempfile.mkdtemp(prefix="chess_vision_"))
    upload_path = temp_dir / "input_image"
    try:
        total = 0
        with upload_path.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Image exceeds the upload limit.")
                output.write(chunk)

        squares = process_uploaded_image(upload_path, temp_dir / "squares")
        board, confidence = predict_squares(squares)
        fen = matrix_to_fen(board)
        analysis = analyze_position(fen)

        low_confidence = [
            {"row": r + 1, "column": c + 1, "piece": board[r][c], "confidence": confidence[r][c]}
            for r in range(8) for c in range(8)
            if confidence[r][c] < MIN_CONFIDENCE
        ]
        return {
            "filename": file.filename,
            "fen": fen,
            "board_matrix": board,
            "confidence_matrix": confidence,
            "low_confidence_squares": low_confidence,
            "chess_analysis": analysis,
        }
    except HTTPException:
        raise
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected prediction failure")
        raise HTTPException(status_code=500, detail="Internal processing error.") from exc
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
