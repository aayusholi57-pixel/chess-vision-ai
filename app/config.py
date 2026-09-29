from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = Path(os.getenv("CHESS_MODEL_PATH", str(MODEL_DIR / "resnet18_chess.pth")))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 20 * 1024 * 1024))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.55"))
BOARD_SIZE = int(os.getenv("BOARD_SIZE", "800"))
ALLOWED_ORIGINS = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "").split(",") if x.strip()]
