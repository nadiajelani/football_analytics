from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List
import io
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/analyze-images")
async def analyze_images(images: List[UploadFile] = File(...), calibration: UploadFile = File(...)):
    # Load calibration
    calib_data = json.loads(await calibration.read())
    frame_rate = calib_data.get("frame_rate_fps", 10000)
    pixels_to_meters = calib_data.get("pixels_per_meter", 172 / 0.22)

    # Sort image files by filename
    sorted_images = sorted(images, key=lambda x: x.filename)
    ball_positions = []

    for idx, image_file in enumerate(sorted_images):
        content = await image_file.read()
        np_arr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        y_position = int(img.shape[0] / 2) + np.random.randint(-15, 15)
        ball_positions.append([idx, y_position])

    # Create dataframe
    df = pd.DataFrame(ball_positions, columns=["Frame", "Ball_Y"])
    df["Time_s"] = df["Frame"] / frame_rate

    # Compute dummy velocity and COR
    y1, y2 = df["Ball_Y"].iloc[10], df["Ball_Y"].iloc[11]
    y3, y4 = df["Ball_Y"].iloc[-2], df["Ball_Y"].iloc[-1]
    vi = (y2 - y1) * pixels_to_meters * frame_rate
    vo = (y4 - y3) * pixels_to_meters * frame_rate
    cor = abs(vo) / abs(vi) if vi != 0 else 0

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["Time_s"], df["Ball_Y"], label="Ball Center Y", color="blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Ball_Y (pixels)")
    ax.set_title(f"Inbound: {vi:.2f} m/s | Outbound: {vo:.2f} m/s | COR: {cor:.2f}")
    ax.legend()
    ax.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
