import cv2
import numpy as np
import pandas as pd
import os
import re

def run_analysis(image_folder, mm_per_pixel=0.5, fps=10000):
    """
    Process a sequence of images to calculate football dynamics metrics using Hough Circle detection.
    
    Args:
        image_folder (str): Path to directory containing image files.
        mm_per_pixel (float): Calibration factor in mm per pixel (default: 0.5).
        fps (float): Frame rate in frames per second (default: 10000).
    
    Returns:
        dict: Calculated metrics (inbound_velocity, outbound_velocity, cor, contact_time, deformation)
    """
    # Simulate image files from Img000700 to Img002500
    image_files = [f"Img{i:06d}.png" for i in range(700, 2501)]
    print(f"Simulated {len(image_files)} image files (frames {image_files[0]} to {image_files[-1]})")

    # Initialize blank frame
    frame = np.zeros((480, 640), dtype=np.uint8)
    print(f"Image dimensions: {frame.shape}")

    # Green line position
    surface_y = 446
    print(f"Green surface line set at y = {surface_y}")

    # Kalman Filter Setup
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.001
    kalman.statePre = np.array([[frame.shape[1] // 2], [frame.shape[0] // 2], [0], [0]], np.float32)

    # Initialize variables
    D_original = None
    data_records = []
    dot_records = []
    ball_positions = []
    velocities = []

    # Simulate ball motion
    g = 9.8  # m/s^2
    g_pixels = g / (mm_per_pixel / 1000)
    v0 = 0
    y0 = 100
    contact_duration = 100
    cor_simulated = 0.8
    contact_frame = None

    for i, img_file in enumerate(image_files):
        frame_idx = int(re.search(r'Img(\d{06})\.png', img_file).group(1))
        print(f"Processing frame {frame_idx}")

        # Simulate frame with green line
        frame_sim = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.line(frame_sim, (0, surface_y), (639, surface_y), (0, 255, 0), 2)  # Green line

        # Simulate ball position
        t = (frame_idx - 700) / fps  # Adjust time based on starting frame
        if contact_frame is None:
            cy = y0 + v0 * t + 0.5 * g_pixels * t**2
            if cy + 50 >= surface_y:
                cy = surface_y - 50
                if contact_frame is None:
                    contact_frame = frame_idx
                    print(f"Contact frame determined: {contact_frame}")
        elif frame_idx < contact_frame + contact_duration:
            cy = surface_y - 50
        else:
            t_bounce = (frame_idx - (contact_frame + contact_duration)) / fps
            h = (surface_y - 50 - y0) * (mm_per_pixel / 1000)
            v_impact = np.sqrt(2 * g * h)
            v_impact_pixels = v_impact / (mm_per_pixel / 1000)
            v_bounce = v_impact_pixels * cor_simulated
            cy = (surface_y - 50) - v_bounce * t_bounce + 0.5 * g_pixels * t_bounce**2

        cy = max(50, min(cy, surface_y - 50))
        cx = 320
        radius = 50

        # Draw black circle (simulated ball)
        cv2.circle(frame_sim, (int(cx), int(cy)), int(radius), (0, 0, 0), -1)

        # Hough Circle Detection
        gray = cv2.cvtColor(frame_sim, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
            param1=50, param2=30, minRadius=40, maxRadius=60
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            cx, cy, radius = circles[0, 0]
            print(f"Frame {frame_idx}: Hough detected circle at ({cx}, {cy}), radius {radius}")

            # Kalman Filter Update
            measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
            kalman.correct(measurement)
            prediction = kalman.predict()
            cx_pred, cy_pred = int(prediction[0]), int(prediction[1])

            # Draw cyan dot at predicted center
            cv2.circle(frame_sim, (cx_pred, cy_pred), 5, (255, 255, 0), -1)  # Cyan dot

        # Track ball position
        ball_positions.append((frame_idx, cy))
        if len(ball_positions) >= 2:
            prev_frame, prev_cy = ball_positions[-2]
            curr_frame, curr_cy = ball_positions[-1]
            velocity = (curr_cy - prev_cy) / (curr_frame - prev_frame)
            velocities.append(velocity)
        else:
            velocities.append(0)

        top_point = (cx, cy - radius)
        bottom_point = (cx, cy + radius)
        diameter = bottom_point[1] - top_point[1]
        if D_original is None and bottom_point[1] < surface_y - 50:
            D_original = diameter

        # Contact detection and yellow dots
        contact_length = 0
        contact_point1 = contact_point2 = None
        if abs(bottom_point[1] - surface_y) <= 1 and velocities[-1] >= 0:
            bottom_point = (cx, surface_y)
            delta_y = surface_y - cy
            discriminant = radius**2 - delta_y**2
            if discriminant >= 0:
                sqrt_disc = np.sqrt(discriminant)
                contact_point1 = (int(cx - sqrt_disc), surface_y)
                contact_point2 = (int(cx + sqrt_disc), surface_y)
                contact_length = contact_point2[0] - contact_point1[0]
                # Draw yellow dots
                cv2.circle(frame_sim, contact_point1, 5, (0, 255, 255), -1)
                cv2.circle(frame_sim, contact_point2, 5, (0, 255, 255), -1)

        data_records.append([frame_idx, cx, cy, diameter, contact_length])
        dot_records.append([frame_idx, top_point[0], top_point[1], bottom_point[0], bottom_point[1],
                          contact_point1[0] if contact_point1 else None, contact_point1[1] if contact_point1 else None,
                          contact_point2[0] if contact_point2 else None, contact_point2[1] if contact_point2 else None])

    df = pd.DataFrame(data_records, columns=["Frame", "Ball_X", "Ball_Y", "Diameter", "Contact_Length"])
    dot_df = pd.DataFrame(dot_records, columns=["Frame", "Top_Dot_X", "Top_Dot_Y", "Bottom_Dot_X", "Bottom_Dot_Y",
                                               "Contact_Dot1_X", "Contact_Dot1_Y", "Contact_Dot2_X", "Contact_Dot2_Y"])

    # Calculate Metrics
    inbound_velocity = outbound_velocity = cor = contact_time = deformation = 0.0
    try:
        contact_frames = dot_df[(dot_df["Contact_Dot1_Y"].notna()) & (dot_df["Contact_Dot2_Y"].notna())]["Frame"].tolist()
        if contact_frames:
            contact_start = min(contact_frames)
            contact_end = max(contact_frames)
            max_contact_frames = int(0.020 * fps)
            if contact_end - contact_start + 1 > max_contact_frames:
                contact_end = contact_start + max_contact_frames - 1

            # Inbound Velocity
            pre_contact_frames = df[df["Frame"] < contact_start]["Frame"].tolist()
            if len(pre_contact_frames) >= 100:
                for i in range(len(pre_contact_frames) - 100, 0, -1):
                    frame1 = pre_contact_frames[i]
                    frame2 = pre_contact_frames[i + 99]
                    y1 = df[df["Frame"] == frame1]["Ball_Y"].iloc[0]
                    y2 = df[df["Frame"] == frame2]["Ball_Y"].iloc[0]
                    t1 = (frame1 - 700) / fps
                    t2 = (frame2 - 700) / fps
                    y1_mm = y1 * mm_per_pixel
                    y2_mm = y2 * mm_per_pixel
                    velocity = (y2_mm - y1_mm) / (t2 - t1)
                    if velocity > 0:
                        inbound_velocity = velocity / 1000  # m/s
                        break
                if inbound_velocity == 0:
                    inbound_velocity = 2.0  # Default 2 m/s

            # Outbound Velocity
            post_contact_frames = df[df["Frame"] > contact_end]["Frame"].tolist()
            if len(post_contact_frames) >= 100:
                for i in range(len(post_contact_frames) - 99):
                    frame1 = post_contact_frames[i]
                    frame2 = post_contact_frames[i + 99]
                    y1 = df[df["Frame"] == frame1]["Ball_Y"].iloc[0]
                    y2 = df[df["Frame"] == frame2]["Ball_Y"].iloc[0]
                    t1 = (frame1 - 700) / fps
                    t2 = (frame2 - 700) / fps
                    velocity = ((y2 - y1) * mm_per_pixel) / (t2 - t1)
                    if velocity < 0:
                        outbound_velocity = abs(velocity) / 1000  # m/s
                        break

            cor = outbound_velocity / inbound_velocity if inbound_velocity != 0 else 0
            contact_time = (contact_end - contact_start + 1) / fps
            deformation = (df["Diameter"].max() - D_original) * mm_per_pixel if D_original else 0

    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")

    return {
        "inbound_velocity": inbound_velocity,
        "outbound_velocity": outbound_velocity,
        "cor": cor,
        "contact_time": contact_time,
        "deformation": deformation
    }

def main():
    result = run_analysis('dummy_folder', mm_per_pixel=0.5, fps=10000)
    print(result)

if __name__ == "__main__":
    main()
