import os
import pandas as pd
import numpy as np
import cv2
from math import sqrt

def run_analysis(image_folder, mm_per_pixel=0.5, fps=30):
    """
    Process a sequence of images to calculate football dynamics metrics and annotate images.
    
    Args:
        image_folder (str): Path to the directory containing .bmp image files.
        mm_per_pixel (float): Calibration factor in mm per pixel (default: 0.5).
        fps (float): Frame rate in frames per second (default: 30).
    
    Returns:
        dict: A dictionary containing the calculated metrics:
              - inbound_velocity (m/s)
              - outbound_velocity (m/s)
              - cor (coefficient of restitution)
              - contact_time (s)
              - deformation (mm)
    """
    # Load image files
    image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg'))])
    if not image_files:
        raise ValueError(f"No supported image files (.bmp, .png, .jpg, .jpeg) found in {image_folder}.")
    print(f"Found {len(image_files)} supported image files in {image_folder}")

    # Read the first image to initialize
    frame = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise ValueError(f"Failed to load the first image: {image_files[0]}")
    print(f"Image dimensions: {frame.shape}")

    # Detect surface line using Hough Transform with adjusted parameters
    edges = cv2.Canny(frame, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=20)
    surface_y = frame.shape[0]  # Default to bottom if no line is found
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y1 - y2) < 10:  # Horizontal line
                surface_y = min(y1, y2)
                break
    print(f"Detected surface at y = {surface_y}")

    # Kalman Filter Setup for Stable Circle Detection
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.02
    kalman.statePre = np.array([[frame.shape[1] // 2], [frame.shape[0] // 2], [0], [0]], np.float32)

    # Initialize variables
    D_original = None
    last_valid_radius = None
    last_valid_contact_points = None
    data_records = []
    dot_records = []
    ball_positions = []  # To track ball center (cy) over time
    contact_frames = []  # To track frames where ball is in contact

    # Process frames
    start_frame = 0
    end_frame = len(image_files)
    for i, img_file in enumerate(image_files):
        if i < start_frame or i >= end_frame:
            continue

        frame = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            print(f"Failed to load frame {i}: {img_file}")
            continue

        # Convert to color for annotations
        frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Adjust HoughCircles parameters for better detection
        circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=50, param2=20, minRadius=10, maxRadius=150)

        prediction = kalman.predict()
        predicted_cx, predicted_cy = int(prediction[0]), int(prediction[1])

        cx, cy, radius = None, None, None
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                cx, cy, radius = circle
                break
            print(f"Frame {i}: Circle detected at ({cx}, {cy}) with radius {radius}")

            if last_valid_radius is not None:
                radius = int(0.85 * last_valid_radius + 0.15 * radius)
            last_valid_radius = radius
            kalman.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
        else:
            print(f"Frame {i}: No circle detected, using predicted position ({predicted_cx}, {predicted_cy})")
            cx, cy = predicted_cx, predicted_cy
            radius = last_valid_radius if last_valid_radius is not None else 50

        ball_positions.append((i, cy))  # Track frame number and y-position

        top_dot_x, top_dot_y = None, None
        bottom_dot_x, bottom_dot_y = None, None
        contact_dot1_x, contact_dot1_y = None, None
        contact_dot2_x, contact_dot2_y = None, None
        diameter = 0
        contact_length = 0

        if radius is not None:
            top_point = (cx, cy - int(radius))
            bottom_point = (cx, cy + int(radius))

            # Adjust contact detection with a margin
            if abs(bottom_point[1] - surface_y) <= 2:  # Within 2 pixels of the surface
                bottom_point = (cx, surface_y)

            top_dot_x, top_dot_y = top_point
            bottom_dot_x, bottom_dot_y = bottom_point

            diameter = bottom_point[1] - top_point[1]
            diameter = max(diameter, 20)
            print(f"Frame {i}: Diameter = {diameter} pixels")

            # Relax the condition for setting D_original
            if D_original is None and bottom_point[1] < surface_y - 20:  # Ball at least 20 pixels above surface
                D_original = diameter
                print(f"Frame {i}: D_original set to {D_original} pixels")

            cv2.circle(frame_color, (cx, cy), int(radius), (255, 0, 0), 1)

            contact_point1 = None
            contact_point2 = None
            if abs(bottom_point[1] - surface_y) <= 2:  # Contact detection with margin
                delta_y = surface_y - cy
                discriminant = radius**2 - delta_y**2
                if discriminant >= 0:
                    sqrt_disc = np.sqrt(discriminant)
                    x1 = int(cx - sqrt_disc)
                    x2 = int(cx + sqrt_disc)
                    contact_point1 = (x1, surface_y)
                    contact_point2 = (x2, surface_y)
                    contact_length = x2 - x1
                    contact_frames.append(i)
                    print(f"Frame {i}: Contact detected! Contact Length: {contact_length} pixels")

                    if last_valid_contact_points is not None:
                        x1_prev, x2_prev = last_valid_contact_points
                        if abs(x1 - x1_prev) > 10 or abs(x2 - x2_prev) > 10:
                            x1, x2 = x1_prev, x2_prev
                    
                    last_valid_contact_points = (x1, x2)

                    contact_dot1_x, contact_dot1_y = contact_point1
                    contact_dot2_x, contact_dot2_y = contact_point2

                    cv2.circle(frame_color, contact_point1, 3, (0, 255, 255), -1)
                    cv2.circle(frame_color, contact_point2, 3, (0, 255, 255), -1)
            else:
                print(f"Frame {i}: No contact (bottom_point[1] = {bottom_point[1]}, surface_y = {surface_y})")

        cv2.line(frame_color, (0, surface_y), (frame.shape[1], surface_y), (0, 255, 0), 1)

        if top_point is not None and bottom_point is not None:
            cv2.circle(frame_color, top_point, 3, (255, 255, 0), -1)
            cv2.circle(frame_color, bottom_point, 3, (255, 255, 0), -1)

        annotated_path = os.path.join(image_folder, f"annotated_frame_{i:04d}.bmp")
        cv2.imwrite(annotated_path, frame_color)

        data_records.append([i, cx, cy, diameter, contact_length])
        dot_records.append([i, top_dot_x, top_dot_y, bottom_dot_x, bottom_dot_y, 
                           contact_dot1_x, contact_dot1_y, contact_dot2_x, contact_dot2_y])

    df = pd.DataFrame(data_records, columns=["Frame", "Ball_X", "Ball_Y", "Diameter", "Contact_Length"])
    dot_df = pd.DataFrame(dot_records, columns=["Frame", 
                                                "Top_Dot_X", "Top_Dot_Y", 
                                                "Bottom_Dot_X", "Bottom_Dot_Y", 
                                                "Contact_Dot1_X", "Contact_Dot1_Y", 
                                                "Contact_Dot2_X", "Contact_Dot2_Y"])

    # Calculate Metrics with Fallbacks
    ball_positions_mm = [(frame, y * mm_per_pixel) for frame, y in ball_positions]
    ball_positions_with_time = [(frame / fps, y) for frame, y in ball_positions_mm]

    # Default values in case of failure
    inbound_velocity = 0.0
    outbound_velocity = 0.0
    cor = 0.0
    contact_time = 0.0
    deformation = 0.0

    try:
        if not contact_frames:
            print("Warning: No contact frames detected. Using default values.")
        else:
            contact_start = min(contact_frames)
            contact_end = max(contact_frames)
            print(f"Contact frames: {contact_frames}, Start: {contact_start}, End: {contact_end}")

            pre_contact = [(t, y) for frame, (t, y) in enumerate(ball_positions_with_time) if frame < contact_start]
            print(f"Pre-contact frames: {len(pre_contact)}")
            if len(pre_contact) >= 2:
                t1, y1 = pre_contact[-2]
                t2, y2 = pre_contact[-1]
                inbound_velocity = abs((y2 - y1) / (t2 - t1))  # mm/s
                print(f"Inbound velocity: {inbound_velocity} mm/s (t1={t1}, y1={y1}, t2={t2}, y2={y2})")
            else:
                print("Warning: Not enough frames before contact to calculate inbound velocity.")

            post_contact = [(t, y) for frame, (t, y) in enumerate(ball_positions_with_time) if frame > contact_end]
            print(f"Post-contact frames: {len(post_contact)}")
            if len(post_contact) >= 2:
                t1, y1 = post_contact[0]
                t2, y2 = post_contact[1]
                outbound_velocity = abs((y2 - y1) / (t2 - t1))  # mm/s
                print(f"Outbound velocity: {outbound_velocity} mm/s (t1={t1}, y1={y1}, t2={t2}, y2={y2})")
            else:
                print("Warning: Not enough frames after contact to calculate outbound velocity.")

            cor = outbound_velocity / inbound_velocity if inbound_velocity != 0 else 0
            contact_time = (contact_end - contact_start + 1) / fps  # in seconds
            print(f"Contact time: {contact_time} s")
            max_diameter = df["Diameter"].max()
            deformation = (max_diameter - D_original) * mm_per_pixel if D_original else 0  # in mm
            print(f"Deformation: {deformation} mm (max_diameter={max_diameter}, D_original={D_original})")

    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        # Return default values if calculation fails

    return {
        "inbound_velocity": inbound_velocity / 1000,  # Convert mm/s to m/s
        "outbound_velocity": outbound_velocity / 1000,  # Convert mm/s to m/s
        "cor": cor,
        "contact_time": contact_time,
        "deformation": deformation
    }
