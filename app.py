import cv2
import numpy as np
import os
import pandas as pd
from scipy.signal import savgol_filter
from flask import Flask, request, render_template_string, send_file, after_this_request
import shutil
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Create temporary directories for uploads and outputs
UPLOAD_FOLDER = tempfile.mkdtemp()
OUTPUT_FOLDER = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure the upload and output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def process_images(image_folder, output_dir):
    try:
        image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.bmp')])
        if not image_files:
            raise ValueError("No image files found")

        total_images = len(image_files)
        logger.info(f"Total images found: {total_images}")

        frame = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise ValueError("Failed to load first image")

        # Surface detection
        edges = cv2.Canny(frame, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
        surface_y = frame.shape[0]
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y1 - y2) < 10:
                    surface_y = min(y1, y2)
                    break
        logger.info(f"Detected surface at y = {surface_y}")

        # Kalman filter setup
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.02
        kalman.statePre = np.array([[frame.shape[1] // 2], [frame.shape[0] // 2], [0], [0]], np.float32)

        data_records = []
        dot_records = []
        last_valid_radius = None
        last_valid_contact_points = None
        D_original = None

        # Dynamically adjust the frame range based on available images
        start_frame = 0  # Start from the first image
        end_frame = total_images  # Process all available images
        logger.info(f"Processing frames from {start_frame} to {end_frame}")

        for i, img_file in enumerate(image_files[start_frame:end_frame]):
            frame = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
            if frame is None:
                logger.warning(f"Failed to load image: {img_file}")
                continue

            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=50, param2=30, minRadius=20, maxRadius=100)

            prediction = kalman.predict()
            cx, cy = int(prediction[0]), int(prediction[1])
            radius = last_valid_radius if last_valid_radius is not None else 50

            if circles is not None:
                for circle in np.uint16(np.around(circles))[0, :]:
                    cx, cy, radius = circle
                    break
                if last_valid_radius is not None:
                    radius = int(0.85 * last_valid_radius + 0.15 * radius)
                last_valid_radius = radius
                kalman.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))

            # Compute top & bottom points
            top_y = cy - int(radius)
            bottom_y = min(cy + int(radius), surface_y)
            diameter = max(bottom_y - top_y, 20)

            contact_length = 0
            contact_points = (None, None, None, None)  # (x1, y1, x2, y2)
            if bottom_y == surface_y:
                delta_y = surface_y - cy
                discriminant = radius**2 - delta_y**2
                if discriminant >= 0:
                    sqrt_disc = np.sqrt(discriminant)
                    x1, x2 = int(cx - sqrt_disc), int(cx + sqrt_disc)
                    contact_length = x2 - x1
                    contact_points = (x1, surface_y, x2, surface_y)

            data_records.append([i + start_frame, cx, cy, diameter, contact_length])
            dot_records.append([i + start_frame, cx, top_y, cx, bottom_y, *contact_points])

        # Save CSVs
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(data_records, columns=["Frame", "Ball_X", "Ball_Y", "Diameter", "Contact_Length"])
        df.to_csv(os.path.join(output_dir, "Drop1.csv"), index=False)

        dot_df = pd.DataFrame(dot_records, columns=["Frame", "Top_Dot_X", "Top_Dot_Y", "Bottom_Dot_X", "Bottom_Dot_Y", "Contact_Dot1_X", "Contact_Dot1_Y", "Contact_Dot2_X", "Contact_Dot2_Y"])
        dot_df.to_csv(os.path.join(output_dir, "Drop1_positions.csv"), index=False)

        return {
            "frames_processed": len(data_records),
            "output": {
                "diameter_csv": "Drop1.csv",
                "positions_csv": "Drop1_positions.csv"
            }
        }
    except Exception as e:
        logger.error(f"Error in process_images: {str(e)}")
        raise

def calculate_metrics(positions_csv_path, mm_per_pixel, fps):
    try:
        # Load the positions CSV
        data = pd.read_csv(positions_csv_path)

        # Calculate the center position (y-coordinate) of the ball
        data["Center_Y"] = (data["Top_Dot_Y"] + data["Bottom_Dot_Y"]) / 2

        # Frame rate and time calculation
        frame_rate = fps  # Use user-provided FPS
        delta_t = 1 / frame_rate  # time between frames in seconds

        # Identify contact frames (where Contact_Dot1_X is not NaN)
        contact_frames = data[data["Contact_Dot1_X"].notna()]
        if contact_frames.empty:
            raise ValueError("No contact frames detected. Please ensure the image sequence includes frames where the ball contacts the surface.")

        # Contact start and end frames
        contact_start_frame = contact_frames["Frame"].iloc[0]
        contact_end_frame = contact_frames["Frame"].iloc[-1]
        contact_time = (contact_end_frame - contact_start_frame + 1) * delta_t  # in seconds

        # Calculate inbound velocity (V_i) just before contact
        pre_contact_frame = contact_start_frame - 1
        pre_pre_contact_frame = pre_contact_frame - 1
        if pre_pre_contact_frame < data["Frame"].min() or pre_contact_frame >= data["Frame"].max():
            raise ValueError("Not enough frames before contact to calculate inbound velocity. Please upload more frames capturing the ball's descent.")

        frame_pre_pre = data[data["Frame"] == pre_pre_contact_frame]
        frame_pre = data[data["Frame"] == pre_contact_frame]
        delta_y_inbound = frame_pre["Center_Y"].iloc[0] - frame_pre_pre["Center_Y"].iloc[0]
        V_i = delta_y_inbound / delta_t  # pixels per second

        # Calculate outbound velocity (V_0) just after leaving the surface
        post_contact_frame = contact_end_frame + 1
        post_post_contact_frame = post_contact_frame + 1
        if post_contact_frame >= data["Frame"].max() or post_post_contact_frame > data["Frame"].max():
            raise ValueError("Not enough frames after contact to calculate outbound velocity. Please upload more frames capturing the ball's rebound.")

        frame_post = data[data["Frame"] == post_contact_frame]
        frame_post_post = data[data["Frame"] == post_post_contact_frame]
        delta_y_outbound = frame_post_post["Center_Y"].iloc[0] - frame_post["Center_Y"].iloc[0]
        V_0 = delta_y_outbound / delta_t  # pixels per second

        # Calculate COR
        cor = abs(V_0) / abs(V_i)

        # Convert velocities to meters per second using user-provided mm per pixel
        pixel_to_meter = mm_per_pixel / 1000  # Convert mm to meters
        inbound_velocity = V_i * pixel_to_meter
        outbound_velocity = V_0 * pixel_to_meter

        # Calculate deformation (maximum change in diameter during contact)
        diameter_data = pd.read_csv(os.path.join(app.config['OUTPUT_FOLDER'], "Drop1.csv"))
        initial_diameter = diameter_data["Diameter"].iloc[0]  # Diameter before contact
        min_diameter = diameter_data[diameter_data["Frame"].isin(contact_frames["Frame"])]["Diameter"].min()
        deformation = (initial_diameter - min_diameter) * pixel_to_meter  # in meters

        return {
            "inbound_velocity": round(inbound_velocity, 2),  # m/s
            "outbound_velocity": round(outbound_velocity, 2),  # m/s
            "cor": round(cor, 4),
            "contact_time": round(contact_time * 1000, 2),  # Convert to milliseconds
            "deformation": round(deformation, 4)  # meters
        }
    except Exception as e:
        logger.error(f"Error in calculate_metrics: {str(e)}")
        raise

# HTML template for the upload page
@app.route('/')
def upload_form():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Football Metrics</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f0f4f8;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            h1 {
                text-align: center;
                color: #2c3e50;
                font-size: 2.5em;
                margin-bottom: 0;
            }
            h3 {
                text-align: center;
                color: #7f8c8d;
                font-size: 1.2em;
                margin-top: 5px;
                margin-bottom: 20px;
            }
            .upload-section {
                background-color: #e6f3f3;
                border: 2px dashed #1abc9c;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
                margin-bottom: 20px;
            }
            .upload-section h4 {
                margin: 0 0 15px 0;
                color: #2c3e50;
                font-size: 1.5em;
            }
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                color: #34495e;
            }
            input[type="file"], input[type="number"] {
                width: 100%;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                box-sizing: border-box;
            }
            input[type="number"] {
                width: 100px;
                display: inline-block;
                margin-left: 10px;
            }
            button {
                display: block;
                width: 100%;
                padding: 12px;
                background-color: #1abc9c;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 1.1em;
                transition: background-color 0.3s;
            }
            button:hover {
                background-color: #16a085;
            }
            .results {
                margin-top: 20px;
                padding: 15px;
                background-color: #fff;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .results h4 {
                margin: 0 0 10px 0;
                color: #2c3e50;
                font-size: 1.3em;
            }
            .results p {
                margin: 5px 0;
                color: #34495e;
            }
            .message {
                margin-top: 15px;
                padding: 10px;
                border-radius: 5px;
                text-align: center;
            }
            .success {
                background-color: #d4edda;
                color: #155724;
            }
            .error {
                background-color: #f8d7da;
                color: #721c24;
            }
            a {
                color: #3498db;
                text-decoration: none;
            }
            a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Football Metrics</h1>
            <h3>AI-Based Inbound Velocity, Deformation & Contact Time Analysis</h3>
            <div class="upload-section">
                <h4>Upload Image Sequence</h4>
                <form action="/upload" method="post" enctype="multipart/form-data">
                    <div class="form-group">
                        <input type="file" name="images" id="images" multiple accept=".bmp" required>
                    </div>
                    <div class="form-group">
                        <label for="mm_per_pixel">mm per Pixel:</label>
                        <input type="number" name="mm_per_pixel" id="mm_per_pixel" step="0.1" value="0.5" required>
                    </div>
                    <div class="form-group">
                        <label for="fps">Frames per Second (FPS):</label>
                        <input type="number" name="fps" id="fps" step="1" value="30" required>
                    </div>
                    <button type="submit">Run Analysis</button>
                </form>
            </div>
            <div class="results">
                <h4>Estimated Results</h4>
                <p><strong>Inbound Velocity:</strong> Not calculated yet</p>
                <p><strong>Outbound Velocity:</strong> Not calculated yet</p>
                <p><strong>Coefficient of Restitution (COR):</strong> Not calculated yet</p>
                <p><strong>Contact Time:</strong> Not calculated yet</p>
                <p><strong>Deformation:</strong> Not calculated yet</p>
            </div>
        </div>
    </body>
    </html>
    '''

# Endpoint to handle image uploads and processing
@app.route('/upload', methods=['POST'])
def upload_images():
    # Clear the upload and output folders before processing new files
    shutil.rmtree(app.config['UPLOAD_FOLDER'], ignore_errors=True)
    shutil.rmtree(app.config['OUTPUT_FOLDER'], ignore_errors=True)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

    # Check if files were uploaded
    if 'images' not in request.files:
        return '''
        <div class="container">
            <div class="message error">No images uploaded. Please select at least one .bmp file.</div>
            <a href="/">Go Back</a>
        </div>
        '''

    files = request.files.getlist('images')
    if not files or all(file.filename == '' for file in files):
        return '''
        <div class="container">
            <div class="message error">No images selected. Please select at least one .bmp file.</div>
            <a href="/">Go Back</a>
        </div>
        '''

    # Get mm per pixel and FPS from the form
    try:
        mm_per_pixel = float(request.form.get('mm_per_pixel', 0.5))
        fps = int(request.form.get('fps', 30))
        if mm_per_pixel <= 0 or fps <= 0:
            raise ValueError("mm per Pixel and FPS must be positive numbers.")
    except ValueError as e:
        return f'''
        <div class="container">
            <div class="message error">Invalid input: {str(e)}</div>
            <a href="/">Go Back</a>
        </div>
        '''

    # Save uploaded files
    for file in files:
        if file and file.filename.endswith('.bmp'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

    try:
        # Process the images
        result = process_images(app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'])
        frames_processed = result['frames_processed']
        positions_csv = os.path.join(app.config['OUTPUT_FOLDER'], result['output']['positions_csv'])

        # Calculate metrics
        metrics = calculate_metrics(positions_csv, mm_per_pixel, fps)

        # Render the page with the results
        return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Football Metrics</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f0f4f8;
                }
                .container {
                    max-width: 800px;
                    margin: 0 auto;
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }
                h1 {
                    text-align: center;
                    color: #2c3e50;
                    font-size: 2.5em;
                    margin-bottom: 0;
                }
                h3 {
                    text-align: center;
                    color: #7f8c8d;
                    font-size: 1.2em;
                    margin-top: 5px;
                    margin-bottom: 20px;
                }
                .upload-section {
                    background-color: #e6f3f3;
                    border: 2px dashed #1abc9c;
                    border-radius: 10px;
                    padding: 20px;
                    text-align: center;
                    margin-bottom: 20px;
                }
                .upload-section h4 {
                    margin: 0 0 15px 0;
                    color: #2c3e50;
                    font-size: 1.5em;
                }
                .form-group {
                    margin-bottom: 15px;
                }
                label {
                    display: block;
                    margin-bottom: 5px;
                    color: #34495e;
                }
                input[type="file"], input[type="number"] {
                    width: 100%;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    box-sizing: border-box;
                }
                input[type="number"] {
                    width: 100px;
                    display: inline-block;
                    margin-left: 10px;
                }
                button {
                    display: block;
                    width: 100%;
                    padding: 12px;
                    background-color: #1abc9c;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 1.1em;
                    transition: background-color 0.3s;
                }
                button:hover {
                    background-color: #16a085;
                }
                .results {
                    margin-top: 20px;
                    padding: 15px;
                    background-color: #fff;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }
                .results h4 {
                    margin: 0 0 10px 0;
                    color: #2c3e50;
                    font-size: 1.3em;
                }
                .results p {
                    margin: 5px 0;
                    color: #34495e;
                }
                .message {
                    margin-top: 15px;
                    padding: 10px;
                    border-radius: 5px;
                    text-align: center;
                }
                .success {
                    background-color: #d4edda;
                    color: #155724;
                }
                .error {
                    background-color: #f8d7da;
                    color: #721c24;
                }
                a {
                    color: #3498db;
                    text-decoration: none;
                }
                a:hover {
                    text-decoration: underline;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Football Metrics</h1>
                <h3>AI-Based Inbound Velocity, Deformation & Contact Time Analysis</h3>
                <div class="message success">
                    Successfully processed {{ frames_processed }} frames.
                </div>
                <div class="upload-section">
                    <h4>Upload Image Sequence</h4>
                    <form action="/upload" method="post" enctype="multipart/form-data">
                        <div class="form-group">
                            <input type="file" name="images" id="images" multiple accept=".bmp" required>
                        </div>
                        <div class="form-group">
                            <label for="mm_per_pixel">mm per Pixel:</label>
                            <input type="number" name="mm_per_pixel" id="mm_per_pixel" step="0.1" value="{{ mm_per_pixel }}" required>
                        </div>
                        <div class="form-group">
                            <label for="fps">Frames per Second (FPS):</label>
                            <input type="number" name="fps" id="fps" step="1" value="{{ fps }}" required>
                        </div>
                        <button type="submit">Run Analysis</button>
                    </form>
                </div>
                <div class="results">
                    <h4>Estimated Results</h4>
                    <p><strong>Inbound Velocity:</strong> {{ metrics.inbound_velocity }} m/s</p>
                    <p><strong>Outbound Velocity:</strong> {{ metrics.outbound_velocity }} m/s</p>
                    <p><strong>Coefficient of Restitution (COR):</strong> {{ metrics.cor }}</p>
                    <p><strong>Contact Time:</strong> {{ metrics.contact_time }} ms</p>
                    <p><strong>Deformation:</strong> {{ metrics.deformation }} m</p>
                </div>
            </div>
        </body>
        </html>
        ''', frames_processed=frames_processed, metrics=metrics, mm_per_pixel=mm_per_pixel, fps=fps)
    except Exception as e:
        logger.error(f"Error in upload_images: {str(e)}")
        return f'''
        <div class="container">
            <div class="message error">Failed to process images. Server returned 500: 'error':'{str(e)}'</div>
            <a href="/">Go Back</a>
        </div>
        ''', 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
