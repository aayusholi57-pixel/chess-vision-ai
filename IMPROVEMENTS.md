# Engineering Upgrade Log

- Removed silent fallback to an untrained/random model.
- Added strict model artifact validation.
- Added absolute model paths.
- Added batched 64-square inference.
- Added confidence scores and low-confidence reporting.
- Added stronger board-corner detection and explicit failure instead of resizing arbitrary images.
- Added 8x8/FEN validation.
- Added class-balanced training.
- Added deterministic seeds.
- Added transfer learning, label smoothing and cosine scheduling.
- Added held-out classification evaluation and confusion matrix.
- Added compilation, Ruff, coverage and Docker CI.
- Added reproducible GitHub Actions training.
- Added model artifacts to Git ignore so large binaries do not pollute source history.

## Remaining empirical gate

The software is structured for reproducible training, but accuracy must come from a real evaluation run. Do not invent performance numbers.
