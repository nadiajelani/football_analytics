import cv2
import numpy as np
import pandas as pd
import os

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
    # Load image files (uncomment for local/Colab environment)
    """
    image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.bmp')])
    if not image_files:
        raise ValueError(f"No .bmp files found in {image_folder}. Please check the folder path and ensure .bmp files exist.")
    print(f"Found {len(image_files)} supported image files in {image_folder}")
    """

    # Placeholder: Assume image_files is a list of pre-loaded image arrays for Pyodide
    image_files = [f"frame_{i:04d}.bmp" for i in range(1450, 1650)]  # Simulated frames 1450 to 1650
    print(f"Simulated {len(image_files)} image files for Pyodide compatibility (frames 1450 to 1650)")

    # Simulate the first image to initialize
    frame = np.zeros((480, 640), dtype=np.uint8)  # Simulated 640x480 grayscale image
    print(f"Image dimensions: {frame.shape}")

    # Set surface_y to the known position of the green line
    surface_y = 446  # As specified, the green line is at y = 446
    print(f"Surface set at y = {surface_y}")

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

    # Process frames 1450 to 1650
    start_frame = 1450
    end_frame = 1650
    for i, img_file in enumerate(image_files):
        frame_idx = i + start_frame  # Adjust frame index to match 1450 to 1650
        if frame_idx < start_frame or frame_idx >= end_frame:
            continue

        # Simulate loading the frame (uncomment for local environment)
        """
        frame = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            print(f"Failed to load {img_file}")
            continue
        """
        frame = np.zeros((480, 640), dtype=np.uint8)  # Simulated grayscale image
        print(f"Processing frame {frame_idx}: {img_file}")

        # Convert to color for annotations
        frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Adjust HoughCircles parameters for better detection
        circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=50, param2=30, minRadius=20, maxRadius=100)

        prediction = kalman.predict()
        predicted_cx, predicted_cy = int(prediction[0]), int(prediction[1])

        cx, cy, radius = None, None, None
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                cx, cy, radius = circle
                break
            print(f"Frame {frame_idx}: Circle detected at ({cx}, {cy}) with radius {radius}")

            if last_valid_radius is not None:
                radius = int(0.85 * last_valid_radius + 0.15 * radius)
            last_valid_radius = radius
            kalman.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
        else:
            print(f"Frame {frame_idx}: No circle detected, using predicted position ({predicted_cx}, {predicted_cy})")
            cx, cy = predicted_cx, predicted_cy
            radius = last_valid_radius if last_valid_radius is not None else 50

        # Track ball position
        ball_positions.append((frame_idx, cy))

        top_dot_x, top_dot_y = None, None
        bottom_dot_x, bottom_dot_y = None, None
        contact_dot1_x, contact_dot1_y = None, None
        contact_dot2_x, contact_dot2_y = None, None
        diameter = 0
        contact_length = 0

        if radius is not None:
            top_point = (cx, cy - int(radius))
            bottom_point = (cx, cy + int(radius))

            if bottom_point[1] >= surface_y:
                bottom_point = (cx, surface_y)

            top_dot_x, top_dot_y = top_point
            bottom_dot_x, bottom_dot_y = bottom_point

            diameter = bottom_point[1] - top_point[1]
            diameter = max(diameter, 20)
            print(f"Frame {frame_idx}: Diameter = {diameter} pixels")

            if D_original is None and bottom_point[1] < surface_y - 50:
                D_original = diameter
                print(f"Frame {frame_idx}: D_original set to {D_original} pixels")

            cv2.circle(frame_color, (cx, cy), int(radius), (255, 0, 0), 1)

            contact_point1 = None
            contact_point2 = None
            if bottom_point[1] == surface_y:
                delta_y = surface_y - cy
                discriminant = radius**2 - delta_y**2
                if discriminant >= 0:
                    sqrt_disc = np.sqrt(discriminant)
                    x1 = int(cx - sqrt_disc)
                    x2 = int(cx + sqrt_disc)
                    contact_point1 = (x1, surface_y)
                    contact_point2 = (x2, surface_y)
                    contact_length = x2 - x1
                    print(f"Frame {frame_idx}: Contact detected! Contact Length: {contact_length} pixels")

                    if last_valid_contact_points is not None:
                        x1_prev, x2_prev = last_valid_contact_points
                        if abs(x1 - x1_prev) > 10 or abs(x2 - x2_prev) > 10:
                            x1, x2 = x1_prev, x2_prev
                    
                    last_valid_contact_points = (x1, x2)

                    contact_dot1_x, contact_dot1_y = contact_point1
                    contact_dot2_x, contact_dot2_y = contact_point2

                    cv2.circle(frame_color, contact_point1, 3, (0, 255, 255), -1)  # Yellow dot (appears cyan in some displays)
                    cv2.circle(frame_color, contact_point2, 3, (0, 255, 255), -1)  # Yellow dot

        cv2.line(frame_color, (0, surface_y), (frame.shape[1], surface_y), (0, 255, 0), 1)  # Green line

        if top_point is not None and bottom_point is not None:
            cv2.circle(frame_color, top_point, 3, (255, 255, 0), -1)  # Cyan dot for diameter
            cv2.circle(frame_color, bottom_point, 3, (255, 255, 0), -1)  # Cyan dot for diameter

        # Note: Pyodide does not support file I/O or cv2_imshow
        # Uncomment the following for Google Colab or local environment
        """
        from google.colab.patches import cv2_imshow
        cv2_imshow(frame_color)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        """

        data_records.append([frame_idx, cx, cy, diameter, contact_length])
        dot_records.append([frame_idx, top_dot_x, top_dot_y, bottom_dot_x, bottom_dot_y, 
                           contact_dot1_x, contact_dot1_y, contact_dot2_x, contact_dot2_y])

    # Create DataFrames (in-memory, no CSV saving)
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
        # Determine contact start and end using yellow dots (Contact_Dot1_Y and Contact_Dot2_Y)
        contact_frames = dot_df[
            (dot_df["Contact_Dot1_Y"].notna()) & (dot_df["Contact_Dot2_Y"].notna())
        ]["Frame"].tolist()

        if not contact_frames:
            print("Warning: No contact frames detected (no yellow dots found). Using default values.")
        else:
            contact_start = min(contact_frames)
            contact_end = max(contact_frames)
            print(f"Contact frames (based on yellow dots): {contact_frames}, Start: {contact_start}, End: {contact_end}")

            # Inbound Velocity: Use frames just before contact_start
            pre_contact_frames = df[df["Frame"] < contact_start]["Frame"].tolist()
            if len(pre_contact_frames) >= 2:
                frame1 = pre_contact_frames[-2]
                frame2 = pre_contact_frames[-1]
                y1 = df[df["Frame"] == frame1]["Ball_Y"].iloc[0]
                y2 = df[df["Frame"] == frame2]["Ball_Y"].iloc[0]
                t1 = frame1 / fps
                t2 = frame2 / fps
                y1_mm = y1 * mm_per_pixel
                y2_mm = y2 * mm_per_pixel
                inbound_velocity = (y2_mm - y1_mm) / (t2 - t1)  # mm/s
                inbound_velocity = abs(inbound_velocity)  # Ensure positive (downward)
                print(f"Inbound velocity: {inbound_velocity} mm/s (Frame {frame1}: y={y1_mm} mm, Frame {frame2}: y={y2_mm} mm)")
                inbound_velocity = inbound_velocity / 1000  # Convert to m/s
            else:
                print("Warning: Not enough frames before contact to calculate inbound velocity.")

            # Outbound Velocity: Use frames just after contact_end
            post_contact_frames = df[df["Frame"] > contact_end]["Frame"].tolist()
            if len(post_contact_frames) >= 2:
                frame1 = post_contact_frames[0]
                frame2 = post_contact_frames[1]
                y1 = df[df["Frame"] == frame1]["Ball_Y"].iloc[0]
                y2 = df[df["Frame"] == frame2]["Ball_Y"].iloc[0]
                t1 = frame1 / fps
                t2 = frame2 / fps
                y1_mm = y1 * mm_per_pixel
                y2_mm = y2 * mm_per_pixel
                outbound_velocity = (y2_mm - y1_mm) / (t2 - t1)  # mm/s
                outbound_velocity = abs(outbound_velocity)  # Ensure positive (upward)
                print(f"Outbound velocity: {outbound_velocity} mm/s (Frame {frame1}: y={y1_mm} mm, Frame {frame2}: y={y2_mm} mm)")
                outbound_velocity = outbound_velocity / 1000  # Convert to m/s
            else:
                print("Warning: Not enough frames after contact to calculate outbound velocity.")

            # Coefficient of Restitution (COR)
            cor = outbound_velocity / inbound_velocity if inbound_velocity != 0 else 0
            print(f"COR: {cor}")

            # Contact Time
            contact_time = (contact_end - contact_start + 1) / fps  # in seconds
            print(f"Contact time: {contact_time} s")

            # Deformation
            max_diameter = df["Diameter"].max()
            deformation = (max_diameter - D_original) * mm_per_pixel if D_original else 0  # in mm
            print(f"Deformation: {deformation} mm (max_diameter={max_diameter}, D_original={D_original})")

    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")

    return {
        "inbound_velocity": inbound_velocity,  # m/s
        "outbound_velocity": outbound_velocity,  # m/s
        "cor": cor,
        "contact_time": contact_time,  # s
        "deformation": deformation  # mm
    }

def main():
    # Simulate running the function with a dummy image folder
    print(">>> run_analysis('dummy_folder', mm_per_pixel=0.5, fps=30)")
    result = run_analysis('dummy_folder', mm_per_pixel=0.5, fps=30)
    print(result)

if __name__ == "__main__":
    main()
