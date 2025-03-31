import os
import pandas as pd
import numpy as np
import cv2
from math import sqrt

def run_analysis(image_folder):
    """
    Process a sequence of images to calculate football dynamics metrics.
    
    Args:
        image_folder (str): Path to the directory containing .bmp image files.
    
    Returns:
        dict: A dictionary containing the calculated metrics:
              - inbound_velocity (m/s)
              - outbound_velocity (m/s)
              - cor (coefficient of restitution)
              - contact_time (s)
              - deformation (mm)
    """
    # Load image files
    image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.bmp')])
    if not image_files:
        raise ValueError(f"No .bmp files found in {image_folder}.")

    # Read the first image to initialize
    frame = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise ValueError(f"Failed to load the first image: {image_files[0]}")

    # Detect surface line using Hough Transform
    edges = cv2.Canny(frame, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    surface_y = frame.shape[0]  # Default to bottom if no line is found
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y1 - y2) < 10:  # Horizontal line
                surface_y = min(y1, y2)

    # Assume a calibration factor (mm per pixel) - replace with actual value
    mm_per_pixel = 0.5  # Example: 1 pixel = 0.5 mm (from previous calibration)

    # Assume frame rate (frames per second) - replace with actual value
    fps = 30  # Example: 30 frames per second

    # Initialize variables
    D_original = None
    last_valid_radius = None
    last_valid_contact_points = None
    data_records = []

    # Process frames (adjust range as needed)
    start_frame = 0
    end_frame = len(image_files)
    ball_positions = []  # To track ball center (cy) over time
    contact_frames = []  # To track frames where ball is in contact

    for i, img_file in enumerate(image_files):
        if i < start_frame or i >= end_frame:
            continue

        frame = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            continue

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=50, param2=30, minRadius=20, maxRadius=100)

        cx, cy, radius = None, None, None
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                cx, cy, radius = circle
                break

            if last_valid_radius is not None:
                radius = int(0.85 * last_valid_radius + 0.15 * radius)
            last_valid_radius = radius

        if radius is None:
            continue

        ball_positions.append((i, cy))  # Track frame number and y-position

        top_point = (cx, cy - int(radius))
        bottom_point = (cx, cy + int(radius))
        if bottom_point[1] >= surface_y:
            bottom_point = (cx, surface_y)

        diameter = bottom_point[1] - top_point[1]
        diameter = max(diameter, 20)

        if D_original is None and bottom_point[1] < surface_y - 50:
            D_original = diameter

        contact_length = 0
        if bottom_point[1] == surface_y:
            delta_y = surface_y - cy
            discriminant = radius**2 - delta_y**2
            if discriminant >= 0:
                sqrt_disc = np.sqrt(discriminant)
                x1 = int(cx - sqrt_disc)
                x2 = int(cx + sqrt_disc)
                contact_length = x2 - x1
                contact_frames.append(i)

        data_records.append([i, cx, cy, diameter, contact_length])

    # Create DataFrame
    df = pd.DataFrame(data_records, columns=["Frame", "Ball_X", "Ball_Y", "Diameter", "Contact_Length"])

    # Calculate Metrics
    # 1. Inbound and Outbound Velocity
    # Convert y-positions to mm
    ball_positions_mm = [(frame, y * mm_per_pixel) for frame, y in ball_positions]
    # Calculate time (in seconds) for each frame
    ball_positions_with_time = [(frame / fps, y) for frame, y in ball_positions_mm]

    # Find contact period
    if not contact_frames:
        raise ValueError("No contact frames detected.")
    contact_start = min(contact_frames)
    contact_end = max(contact_frames)

    # Inbound velocity: before contact
    pre_contact = [(t, y) for frame, (t, y) in enumerate(ball_positions_with_time) if frame < contact_start]
    if len(pre_contact) < 2:
        raise ValueError("Not enough frames before contact to calculate inbound velocity.")
    t1, y1 = pre_contact[-2]
    t2, y2 = pre_contact[-1]
    inbound_velocity = abs((y2 - y1) / (t2 - t1))  # m/s (assuming mm/s converted to m/s)

    # Outbound velocity: after contact
    post_contact = [(t, y) for frame, (t, y) in enumerate(ball_positions_with_time) if frame > contact_end]
    if len(post_contact) < 2:
        raise ValueError("Not enough frames after contact to calculate outbound velocity.")
    t1, y1 = post_contact[0]
    t2, y2 = post_contact[1]
    outbound_velocity = abs((y2 - y1) / (t2 - t1))  # m/s

    # 2. Coefficient of Restitution (COR)
    cor = outbound_velocity / inbound_velocity if inbound_velocity != 0 else 0

    # 3. Contact Time
    contact_time = (contact_end - contact_start + 1) / fps  # in seconds

    # 4. Deformation
    max_diameter = df["Diameter"].max()
    deformation = (max_diameter - D_original) * mm_per_pixel if D_original else 0  # in mm

    # Return results in the expected format
    return {
        "inbound_velocity": inbound_velocity / 1000,  # Convert mm/s to m/s
        "outbound_velocity": outbound_velocity / 1000,  # Convert mm/s to m/s
        "cor": cor,
        "contact_time": contact_time,
        "deformation": deformation
    }
