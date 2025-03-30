import cv2
import numpy as np
import pandas as pd
import os

# This function will be called by Flask API

def run_analysis(image_folder):
    image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.bmp')])

    if not image_files:
        raise ValueError("No BMP images found in uploaded files.")

    # Kalman filter setup (simplified for processing)
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    data_records = []
    last_valid_radius = None
    surface_y = None

    for i, img_file in enumerate(image_files):
        frame = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            continue

        if surface_y is None:
            edges = cv2.Canny(frame, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
            surface_y = frame.shape[0]
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if abs(y1 - y2) < 10:
                        surface_y = min(y1, y2)
                        break

        circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1.2, 50, param1=50, param2=30, minRadius=20, maxRadius=100)

        prediction = kalman.predict()
        predicted_cx, predicted_cy = int(prediction[0]), int(prediction[1])

        cx, cy, radius = predicted_cx, predicted_cy, 50
        if circles is not None:
            circles = np.uint16(np.around(circles))
            cx, cy, radius = circles[0][0]
            if last_valid_radius is not None:
                radius = int(0.85 * last_valid_radius + 0.15 * radius)
            last_valid_radius = radius
            kalman.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))

        top_y = cy - radius
        bottom_y = min(cy + radius, surface_y)
        diameter = bottom_y - top_y
        contact_length = 0

        if bottom_y == surface_y:
            delta_y = surface_y - cy
            if radius**2 - delta_y**2 >= 0:
                contact_length = int(2 * np.sqrt(radius**2 - delta_y**2))

        data_records.append([i + 1, cx, cy, diameter, contact_length])

    df = pd.DataFrame(data_records, columns=["Frame", "Ball_X", "Ball_Y", "Diameter", "Contact_Length"])
    df = df.dropna()

    frame_rate = 10000
    time_interval = 1 / frame_rate
    pixels_to_meters = 0.00128

    df["Time_s"] = (df["Frame"] - df["Frame"].min()) / frame_rate

    try:
        y_1534 = df.loc[df["Frame"] == 1534, "Ball_Y"].values[0]
        y_1535 = df.loc[df["Frame"] == 1535, "Ball_Y"].values[0]
        vi_pixels = (y_1535 - y_1534) / time_interval
        vi_mps = vi_pixels * pixels_to_meters
    except:
        vi_mps = 0

    try:
        y_1640 = df.loc[df["Frame"] == 1640, "Ball_Y"].values[0]
        y_1641 = df.loc[df["Frame"] == 1641, "Ball_Y"].values[0]
        vo_pixels = (y_1641 - y_1640) / time_interval
        vo_mps = vo_pixels * pixels_to_meters
    except:
        vo_mps = 0

    cor = abs(vo_mps / vi_mps) if vi_mps != 0 else 0
    contact_time = (df["Contact_Length"] > 0).sum() * time_interval
    deformation = df["Diameter"].max() - df["Diameter"].min()

    return {
        "inbound_velocity": round(vi_mps, 2),
        "outbound_velocity": round(vo_mps, 2),
        "cor": round(cor, 3),
        "contact_time": round(contact_time, 4),
        "deformation": round(deformation, 2)
    }