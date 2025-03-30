from flask import Flask, request, jsonify
from analysis import run_analysis  # your logic here
import tempfile, os, shutil

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    files = request.files.getlist('images')
    temp_dir = tempfile.mkdtemp()

    try:
        for i, file in enumerate(files):
            file.save(os.path.join(temp_dir, f"frame_{i:04d}.bmp"))

        result = run_analysis(temp_dir)  # Your actual function
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        shutil.rmtree(temp_dir)

if __name__ == '__main__':
    app.run(debug=True)
