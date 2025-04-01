import os
from flask import Flask, request, jsonify, send_from_directory
from analysis import run_analysis
import tempfile
import shutil

app = Flask(__name__)

# Serve the index.html file at the root URL
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    files = request.files.getlist('images')
    print(f"Received {len(files)} files: {[file.filename for file in files]}")

    # Get mm_per_pixel and fps from the request (if provided)
    mm_per_pixel = float(request.form.get('mm_per_pixel', 0.5))  # Default to 0.5
    fps = float(request.form.get('fps', 30))  # Default to 30
    print(f"Using mm_per_pixel={mm_per_pixel}, fps={fps}")

    temp_dir = tempfile.mkdtemp()

    try:
        for i, file in enumerate(files):
            file.save(os.path.join(temp_dir, f"frame_{i:04d}.bmp"))

        result = run_analysis(temp_dir, mm_per_pixel=mm_per_pixel, fps=fps)
        print(f"Analysis result: {result}")
        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()  # This prints full error to logs
        return jsonify({'error': str(e)}), 500
    finally:
        shutil.rmtree(temp_dir)

# ✅ Bind to 0.0.0.0 and use the PORT Render provides
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(debug=True, host='0.0.0.0', port=port)
