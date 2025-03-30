from flask import Flask, request, jsonify
from analysis import run_analysis
import tempfile, os, shutil

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    files = request.files.getlist('images')
    temp_dir = tempfile.mkdtemp()

    try:
        for i, file in enumerate(files):
            file.save(os.path.join(temp_dir, f"frame_{i:04d}.bmp"))

        result = run_analysis(temp_dir)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        shutil.rmtree(temp_dir)

# ✅ Bind to 0.0.0.0 and use the PORT Render provides
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
