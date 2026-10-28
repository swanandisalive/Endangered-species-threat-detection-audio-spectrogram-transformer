"""
app.py
FastAPI server with WebSocket audio streaming for real-time AST 16x16 inference.
"""

import asyncio
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from inference_engine import SAMPLE_RATE, TARGET_SAMPLES, LiveASTPredictor

app = FastAPI(title="EcoAST Live Acoustic Monitor")

# Load model once at startup (automatically checks root and trained_models/)
predictor = LiveASTPredictor(
    weights_path="best_ast_faststride_model.pth",
    confidence_threshold=0.35,
    rms_silence_threshold=0.008,
)

if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    static_index = Path("static/index.html")
    if static_index.exists():
        return FileResponse(static_index)
    return FileResponse("index.html")


@app.websocket("/ws/audio")
async def audio_stream(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Microphone stream connected.")

    rolling_buffer = np.zeros(TARGET_SAMPLES, dtype=np.float32)
    samples_received = 0
    min_samples_for_first_pred = SAMPLE_RATE * 2  # Start after 2s of audio
    last_inference_time = 0.0
    last_log_time = (
        time.time()
    )  # Initialize to current time so first log happens at +15s

    inference_interval = 0.5  # Real-time UI updates for probabilities & status
    log_interval = 15.0  # Create event log entries strictly every 15 seconds

    try:
        while True:
            message = await websocket.receive()

            if message.get("bytes"):
                chunk = np.frombuffer(message["bytes"], dtype=np.float32)
                chunk_len = len(chunk)

                if chunk_len > 0:
                    if chunk_len >= TARGET_SAMPLES:
                        rolling_buffer[:] = chunk[-TARGET_SAMPLES:]
                    else:
                        rolling_buffer = np.roll(rolling_buffer, -chunk_len)
                        rolling_buffer[-chunk_len:] = chunk

                    samples_received += chunk_len

                now = time.time()
                if (
                    samples_received >= min_samples_for_first_pred
                    and (now - last_inference_time) >= inference_interval
                ):
                    last_inference_time = now

                    # Run model inference in worker thread
                    result = await asyncio.to_thread(
                        predictor.predict, rolling_buffer.copy()
                    )
                    result["buffer_fill"] = min(
                        1.0, round(samples_received / TARGET_SAMPLES, 2)
                    )

                    # Trigger log entry every 15 seconds
                    if (now - last_log_time) >= log_interval:
                        result["should_log"] = True
                        last_log_time = now
                    else:
                        result["should_log"] = False

                    result["next_log_in"] = max(
                        0.0, round(log_interval - (now - last_log_time), 1)
                    )

                    await websocket.send_json(result)

            elif "text" in message and message["text"] == "reset":
                rolling_buffer.fill(0.0)
                samples_received = 0
                last_log_time = time.time()

    except WebSocketDisconnect:
        print("[WS] Microphone stream disconnected.")
