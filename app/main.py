from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import tempfile
import shutil
from pathlib import Path

# Import our custom modules
from app.board import reconstruct_board_from_squares, load_chess_model
from app.fen import matrix_to_fen
from app.preprocessing import process_uploaded_image
from app.chess_engine import analyze_position

app = FastAPI(title="Chess Vision AI", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    _, _, is_trained = load_chess_model()
    model_status = "trained" if is_trained else "untrained (predictions will be unreliable)"
    return {"status": "healthy", "model": model_status}

@app.post("/predict")
async def predict_board(file: UploadFile = File(...)):
    """
    Accepts a chessboard image, processes it, and returns FEN notation + analysis.
    
    Note: Predictions require a trained model. If using an untrained model,
    all predictions will be random. Train your model and save to models/resnet18_chess.pth
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Create a secure temporary directory
    temp_dir = tempfile.mkdtemp(prefix="chess_vision_")
    temp_image_path = None
    
    try:
        # Secure file path handling
        temp_image_path = os.path.join(temp_dir, "upload.jpg")
        
        # Save uploaded file
        with open(temp_image_path, "wb") as buffer:
            content = await file.read()
            if len(content) > 50 * 1024 * 1024:  # 50MB limit
                raise HTTPException(status_code=413, detail="File too large (max 50MB)")
            buffer.write(content)
        
        # 1. OpenCV Preprocessing
        squares_dir = process_uploaded_image(temp_image_path, output_dir=os.path.join(temp_dir, "squares"))
        
        # 2. PyTorch Inference
        matrix = reconstruct_board_from_squares(squares_dir=squares_dir)
        
        # 3. FEN Generation
        fen_string = matrix_to_fen(matrix)
        
        # 4. Chess Engine Analysis
        engine_analysis = analyze_position(fen_string)
        
        # 5. Check model training status
        _, _, is_trained = load_chess_model()
        
        return {
            "filename": file.filename,
            "fen": fen_string,
            "engine_analysis": engine_analysis,
            "board_matrix": matrix,
            "warning": "Model is untrained. Predictions are unreliable." if not is_trained else None
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        # Guaranteed cleanup of temporary files
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
