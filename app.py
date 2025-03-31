from flask import Flask, request, jsonify
from analysis import run_analysis
import tempfile, os, shutil

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    files = request.files.getlist('images')
    print(f"Received {len(files)} files: {[file.filename for file in files]}")

    temp_dir = tempfile.mkdtemp()
    print(f"Temporary directory: {temp_dir}")  # Log the temp directory path

    try:
        for i, file in enumerate(files):
            file.save(os.path.join(temp_dir, f"frame_{i:04d}.bmp"))

        result = run_analysis(temp_dir)
        print(f"Analysis result: {result}")
        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        # Optionally copy annotated images to a persistent directory for debugging
        # import shutil
        # persistent_dir = "/path/to/save/annotated/images"
        # if not os.path.exists(persistent_dir):
        #     os.makedirs(persistent_dir)
        # for file in os.listdir(temp_dir):
        #     if file.startswith("annotated_frame_"):
        #         shutil.copy(os.path.join(temp_dir, file), persistent_dir)
        shutil.rmtree(temp_dir)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(debug=False, host='0.0.0.0', port=port)
