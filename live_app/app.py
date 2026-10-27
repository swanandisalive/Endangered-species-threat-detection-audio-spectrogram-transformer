"""
app.py
FastAPI server with WebSocket audio streaming for real-time AST 16x16 inference.
"""

import asyncio
import time

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from inference_engine import SAMPLE_RATE, TARGET_SAMPLES, LiveASTPredictor

app = FastAPI(title="EcoAST Live Acoustic Monitor")

# Load the model once at startup
predictor = LiveASTPredictor(
    weights_path="trained_models/5_classes_best_ast_faststride_model.pth",
    confidence_threshold=0.35,
    rms_silence_threshold=0.008,
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.websocket("/ws/audio")
async def audio_stream(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Microphone stream connected.")

    # Rolling 10.24s buffer (163,840 float32 samples at 16kHz)
    rolling_buffer = np.zeros(TARGET_SAMPLES, dtype=np.float32)
    samples_received = 0
    min_samples_for_first_pred = SAMPLE_RATE * 2  # Start predicting after 2s of audio
    last_inference_time = 0.0
    inference_interval = 0.8  # Run inference every 0.8 seconds

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
                    # Run PyTorch inference in a worker thread to keep WebSocket responsive
                    result = await asyncio.to_thread(
                        predictor.predict, rolling_buffer.copy()
                    )
                    result["buffer_fill"] = min(
                        1.0, round(samples_received / TARGET_SAMPLES, 2)
                    )
                    await websocket.send_json(result)

            elif "text" in message and message["text"] == "reset":
                rolling_buffer.fill(0.0)
                samples_received = 0

    except WebSocketDisconnect:
        print("[WS] Microphone stream disconnected.")
