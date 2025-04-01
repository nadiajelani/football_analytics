import os
import cv2
import numpy as np  # Add this import for cv2.imdecode
from flask import Flask, request, jsonify, send_file
from analysis import run_analysis
import shutil
import tempfile

app = Flask(__name__)

@app.route('/')
def index():
    print("Serving index.html")
    return send_file('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    print("Received request to /analyze")
    if 'images' not in request.files:
        return jsonify({'error': 'No images uploaded'}), 400

    files = request.files.getlist('images')
    print(f"Received {len(files)} files: {[f.filename for f in files]}")

    mm_per_pixel = float(request.form.get('mm_per_pixel', 0.5))
    fps = float(request.form.get('fps', 30))
    print(f"Using mm_per_pixel={mm_per_pixel}, fps={fps}")

    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")

    max_dim = 320
    for i, file in enumerate(files):
        if file:
            img_path = os.path.join(temp_dir, f"frame_{i:04d}.jpg")
            # Read the file into a NumPy array
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            if img is None:
                shutil.rmtree(temp_dir)
                return jsonify({'error': f'Failed to load image {file.filename}'}), 400

            height, width = img.shape[:2]
            scale = min(max_dim / width, max_dim / height)
            if scale < 1:
                img = cv2.resize(img, (int(width * scale), int(height * scale)))
                print(f"Resized image {file.filename} to {img.shape[1]}x{img.shape[0]}")
            cv2.imwrite(img_path, img)
            print(f"Saved file: {img_path}")

    try:
        result = run_analysis(temp_dir, mm_per_pixel=mm_per_pixel, fps=fps)
        print(f"Analysis result: {result}")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)

    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
