import os
from flask import Flask, request, jsonify, send_from_directory
from analysis import run_analysis
import tempfile
import shutil
import cv2
import numpy as np

app = Flask(__name__)

# Serve the index.html file at the root URL
@app.route('/')
def serve_index():
    print("Serving index.html")
    return send_from_directory('.', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    print("Received request to /analyze")
    files = request.files.getlist('images')
    print(f"Received {len(files)} files: {[file.filename for file in files]}")

    if not files:
        print("No files received in the request")
        return jsonify({'error': 'No image files provided'}), 400

    # Get mm_per_pixel and fps from the request (if provided)
    mm_per_pixel = float(request.form.get('mm_per_pixel', 0.5))  # Default to 0.5
    fps = float(request.form.get('fps', 30))  # Default to 30
    print(f"Using mm_per_pixel={mm_per_pixel}, fps={fps}")

    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")

    try:
        for i, file in enumerate(files):
            # Read the image using OpenCV
            file_stream = file.read()
            nparr = np.frombuffer(file_stream, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to decode image: {file.filename}")
                continue

            # Resize the image to reduce processing load (e.g., max dimension 640)
            max_dim = 640
            height, width = img.shape
            if max(height, width) > max_dim:
                scale = max_dim / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                print(f"Resized image {file.filename} to {new_width}x{new_height}")

            # Save the resized image
            ext = os.path.splitext(file.filename)[1]  # e.g., .bmp, .png, .jpg
            file_path = os.path.join(temp_dir, f"frame_{i:04d}{ext}")
            cv2.imwrite(file_path, img)
            print(f"Saved file: {file_path}")

        result = run_analysis(temp_dir, mm_per_pixel=mm_per_pixel, fps=fps)
        print(f"Analysis result: {result}")
        return jsonify(result)

    except Exception as e:
        import traceback
        print("Error during analysis:")
        traceback.print_exc()  # This prints full error to logs
        return jsonify({'error': str(e)}), 500
    finally:
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)

# ✅ Bind to 0.0.0.0 and use the PORT Render provides
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f"Starting Flask app on port {port}")
    app.run(debug=True, host='0.0.0.0', port=port)
