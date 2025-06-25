# app.py
from flask import Flask, render_template, request, redirect
import os
from classification_module import process_data

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PLOTS_FOLDER = 'static/plots'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(file_path)

        metrics, cm_path, roc_path = process_data(file_path, PLOTS_FOLDER)
        return render_template('result.html', metrics=metrics, cm_path=cm_path, roc_path=roc_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
