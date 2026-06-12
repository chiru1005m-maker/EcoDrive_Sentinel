import os
import re
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Determine absolute path to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_FILE = PROJECT_ROOT / "logs" / "train_universal.log"
PUBLIC_DIR = Path(__file__).parent / "public"

app = FastAPI(title="EcoDrive Dashboard Server")

# Serve static files
app.mount("/static", StaticFiles(directory=str(PUBLIC_DIR)), name="static")

@app.get("/")
async def serve_index():
    return FileResponse(str(PUBLIC_DIR / "index.html"))

@app.get("/api/logs")
async def get_training_logs():
    """Parse the train_universal.log file and extract epoch metrics."""
    metrics = []
    try:
        if not LOG_FILE.exists():
            return {"status": "waiting", "data": []}
            
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()[-1000:] # read last 1000 lines
            
        # Regex to match: | epoch | train_loss | val_mae | lr | time
        # Example: 2026-05-29 12:56:08.599 | INFO     | train_universal:300 |      1       0.00166     0.00840    9.99e-04    37.6s
        pattern = re.compile(r"\|\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+[\d.e+-]+\s+[\d.]+s")
        
        unique_metrics = {}
        for line in lines:
            match = pattern.search(line)
            if match:
                ep = int(match.group(1))
                # If we see Epoch 1, a new training run has started, so clear previous metrics
                if ep == 1:
                    unique_metrics.clear()
                    
                unique_metrics[ep] = {
                    "epoch": ep,
                    "train_mse": float(match.group(2)),
                    "val_mse": float(match.group(3))
                }
        
        final_metrics = list(unique_metrics.values())
        final_metrics.sort(key=lambda x: x['epoch'])
        
        return {"status": "success", "data": final_metrics}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Run dashboard server on port 8080 to not conflict with the main API on 8000
    uvicorn.run(app, host="0.0.0.0", port=8080)
