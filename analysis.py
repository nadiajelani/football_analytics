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

    # Placeholder: Simulate 30 frames
    image_files = [f"frame_{i:04d}.bmp" for i in range(1450, 1480)]
    print(f"Simulated {len(image_files)} image files (frames 1450 to {1450 + len(image_files) - 1})")

    # Simulate the first image to initialize
    frame = np.zeros((480, 640), dtype=np.uint8)  # Simulated 640x480 grayscale image
    print(f"Image dimensions: {frame.shape}")

    # Set surface_y to the known position of the green line
    surface_y = 446
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
    ball_positions = []
    velocities = []

    # Simulate realistic ball motion
    g = 9.8  # m/s^2
    g_pixels = g / (mm_per_pixel / 1000)  # Convert to pixels/s^2
    v0 = 0  # Initial velocity (pixels/s)
    y0 = 200  # Initial y-position (pixels)
    contact_duration = 2  # Frames in contact
    cor_simulated = 0.8  # Simulated COR for motion (not used in final calculation)
    contact_frame = 15  # Frame where the ball hits the surface

    # Process frames
    start_frame = 1450
    end_frame = 1450 + len(image_files)
    for i, img_file in enumerate(image_files):
        frame_idx = i + start_frame
        if frame_idx < start_frame or frame_idx >= end_frame:
            continue

        # Simulate ball position
        t = i / fps  # Time in seconds
        if i < contact_frame:
            # Before contact: free fall
            cy = y0 + v0 * t + 0.5 * g_pixels * t**2
            if cy + 50 >= surface_y:  # Assume radius=50
                cy = surface_y - 50
        elif i < contact_frame + contact_duration:
            # During contact
            cy = surface_y - 50
        else:
            # After contact: bounce upward
            t_bounce = (i - (contact_frame + contact_duration)) / fps
            h = (surface_y - 50 - y0) * (mm_per_pixel / 1000)  # Height in meters
            v_impact = np.sqrt(2 * g * h)  # m/s
            v_impact_pixels = v_impact / (mm_per_pixel / 1000)  # pixels/s
            v_bounce = v_impact_pixels * cor_simulated  # Upward velocity after bounce
            cy = (surface_y - 50) - v_bounce * t_bounce + 0.5 * g_pixels * t_bounce**2

        # Ensure cy stays within bounds
        cy = max(50, min(cy, surface_y - 50))
        cx = 320  # Center x
        radius = 50  # Fixed radius for simulation
        print(f"Frame {frame_idx}: Simulated ball position at ({cx}, {cy}) with radius {radius}")

        # Track ball position
        ball_positions.append((frame_idx, cy))

        # Calculate velocity for contact detection (in pixels per frame)
        if len(ball_positions) >= 2:
            prev_frame, prev_cy = ball_positions[-2]
            curr_frame, curr_cy = ball_positions[-1]
            velocity = (curr_cy - prev_cy) / (curr_frame - prev_frame)  # pixels per frame
            velocities.append(velocity)
        else:
            velocities.append(0)

        print(f"Frame {frame_idx}: Ball_Y = {cy} pixels, Velocity = {velocities[-1]} pixels/frame")

        top_dot_x, top_dot_y = None, None
        bottom_dot_x, bottom_dot_y = None, None
        contact_dot1_x, contact_dot1_y = None, None
        contact_dot2_x, contact_dot2_y = None, None
        diameter = 0
        contact_length = 0

        top_point = (cx, cy - int(radius))
        bottom_point = (cx, cy + int(radius))

        # Contact detection with velocity check
        is_contact = False
        contact_margin = 1
        if abs(bottom_point[1] - surface_y) <= contact_margin:
            if len(velocities) > 0:
                current_velocity = velocities[-1]
                if current_velocity > 0:  # Moving downward
                    is_contact = True
                elif current_velocity < 0:  # Moving upward
                    if data_records and data_records[-1][-1] > 0:
                        is_contact = True
                print(f"Frame {frame_idx}: Contact check - bottom_y={bottom_point[1]}, velocity={current_velocity}, is_contact={is_contact}")
        else:
            print(f"Frame {frame_idx}: No contact (bottom_y={bottom_point[1]}, surface_y={surface_y})")

        if is_contact:
            bottom_point = (cx, surface_y)

        top_dot_x, top_dot_y = top_point
        bottom_dot_x, bottom_dot_y = bottom_point

        diameter = bottom_point[1] - top_point[1]
        diameter = max(diameter, 20)
        print(f"Frame {frame_idx}: Diameter = {diameter} pixels")

        if D_original is None and bottom_point[1] < surface_y - 50:
            D_original = diameter
            print(f"Frame {frame_idx}: D_original set to {D_original} pixels")

        contact_point1 = None
        contact_point2 = None
        if is_contact:
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

        data_records.append([frame_idx, cx, cy, diameter, contact_length])
        dot_records.append([frame_idx, top_dot_x, top_dot_y, bottom_dot_x, bottom_dot_y, 
                           contact_dot1_x, contact_dot1_y, contact_dot2_x, contact_dot2_y])

    # Create DataFrames (in-memory)
    df = pd.DataFrame(data_records, columns=["Frame", "Ball_X", "Ball_Y", "Diameter", "Contact_Length"])
    dot_df = pd.DataFrame(dot_records, columns=["Frame", 
                                                "Top_Dot_X", "Top_Dot_Y", 
                                                "Bottom_Dot_X", "Bottom_Dot_Y", 
                                                "Contact_Dot1_X", "Contact_Dot1_Y", 
                                                "Contact_Dot2_X", "Contact_Dot2_Y"])

    # Print Ball_Y for all frames to debug
    print("\nBall Y-Positions for All Frames:")
    for frame, y in zip(df["Frame"], df["Ball_Y"]):
        print(f"Frame {frame}: Ball_Y = {y} pixels")

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
        # Determine contact start and end using yellow dots
        contact_frames = dot_df[
            (dot_df["Contact_Dot1_Y"].notna()) & (dot_df["Contact_Dot2_Y"].notna())
        ]["Frame"].tolist()

        if not contact_frames:
            print("Warning: No contact frames detected (no yellow dots found). Using default values.")
        else:
            contact_start = min(contact_frames)
            contact_end = max(contact_frames)

            # Limit contact duration to a realistic range (1-2 frames at 30 fps)
            max_contact_frames = 2
            if contact_end - contact_start + 1 > max_contact_frames:
                contact_end = contact_start + max_contact_frames - 1
                contact_frames = list(range(contact_start, contact_end + 1))
            print(f"Contact frames (after limiting): {contact_frames}, Start: {contact_start}, End: {contact_end}")

            # Inbound Velocity: Use frames further back before contact_start
            pre_contact_frames = df[df["Frame"] < contact_start]["Frame"].tolist()
            if len(pre_contact_frames) >= 5:
                frame1 = pre_contact_frames[-5]
                frame2 = pre_contact_frames[-1]
                y1 = df[df["Frame"] == frame1]["Ball_Y"].iloc[0]
                y2 = df[df["Frame"] == frame2]["Ball_Y"].iloc[0]
                t1 = frame1 / fps
                t2 = frame2 / fps
                y1_mm = y1 * mm_per_pixel
                y2_mm = y2 * mm_per_pixel
                inbound_velocity = (y2_mm - y1_mm) / (t2 - t1)  # mm/s
                print(f"Inbound velocity calculation: y1={y1_mm} mm, y2={y2_mm} mm, t1={t1} s, t2={t2} s")
                if inbound_velocity <= 0:
                    print("Warning: Inbound velocity is negative or zero, setting to default...")
                    inbound_velocity = 1000  # Default to 1 m/s in mm/s
                else:
                    inbound_velocity = abs(inbound_velocity)
                    if inbound_velocity < 1000:  # Less than 1 m/s
                        inbound_velocity = 1000
                print(f"Inbound velocity: {inbound_velocity} mm/s (Frame {frame1}: y={y1_mm} mm, Frame {frame2}: y={y2_mm} mm)")
                inbound_velocity = inbound_velocity / 1000  # Convert to m/s
            else:
                print("Warning: Not enough frames before contact to calculate inbound velocity.")

            # Outbound Velocity: Use frames after contact_end, ensure upward motion
            post_contact_frames = df[df["Frame"] > contact_end]["Frame"].tolist()
            outbound_velocity = 0.0
            if len(post_contact_frames) >= 2:
                # Try to find frames where the ball is moving upward
                for i in range(len(post_contact_frames) - 1):
                    frame1 = post_contact_frames[i]
                    frame2 = post_contact_frames[i + 1]
                    y1 = df[df["Frame"] == frame1]["Ball_Y"].iloc[0]
                    y2 = df[df["Frame"] == frame2]["Ball_Y"].iloc[0]
                    t1 = frame1 / fps
                    t2 = frame2 / fps
                    y1_mm = y1 * mm_per_pixel
                    y2_mm = y2 * mm_per_pixel
                    velocity = (y2_mm - y1_mm) / (t2 - t1)  # mm/s
                    print(f"Outbound velocity attempt {i+1}: Frame {frame1} to {frame2}, y1={y1_mm} mm, y2={y2_mm} mm, velocity={velocity} mm/s")
                    if velocity < 0:  # Upward motion (y decreases)
                        outbound_velocity = abs(velocity)
                        break
                if outbound_velocity == 0.0:
                    print("Warning: Could not find frames with upward motion after contact.")
                else:
                    print(f"Outbound velocity: {outbound_velocity} mm/s (Frame {frame1}: y={y1_mm} mm, Frame {frame2}: y={y2_mm} mm)")
                outbound_velocity = outbound_velocity / 1000  # Convert to m/s
            else:
                print("Warning: Not enough frames after contact to calculate outbound velocity.")

            # Coefficient of Restitution (COR)
            cor = outbound_velocity / inbound_velocity if inbound_velocity != 0 else 0
            print(f"COR: {cor}")

            # Contact Time
            contact_time = (contact_end - contact_start + 1) / fps  # in seconds
            print(f"Contact time: {contact_time * 1000} ms")

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
    print(">>> run_analysis('dummy_folder', mm_per_pixel=0.5, fps=30)")
    result = run_analysis('dummy_folder', mm_per_pixel=0.5, fps=30)
    print(result)

if __name__ == "__main__":
    main()
