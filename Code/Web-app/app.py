# app.py
import os
from flask import Flask, render_template, request

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'xlsx'}

# Load the saved CNN model
model = load_model('cnn.h5')

# Helper function to check if the file has a valid extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for uploading the experiment data file
@app.route('/upload_experiment', methods=['POST'])
def upload_experiment():
    if 'experiment_file' not in request.files:
        return "No file part"
    file = request.files['experiment_file']
    if file.filename == '':
        return "No selected file"
    if file and allowed_file(file.filename):
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        # Process the experiment file (add your processing logic here)
        return "Experiment data uploaded successfully"

# Route for uploading the vibration data files
@app.route('/upload_vibration', methods=['POST'])
def upload_vibration():
    files = request.files.getlist('vibration_files')
    for file in files:
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            # Process the vibration file (add your processing logic here)
    return "Vibration data uploaded successfully"

# Route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Load the data from the uploaded files (add your loading logic here)
    data = pd.read_excel(os.path.join(app.config['UPLOAD_FOLDER'], 'your_file_name.xlsx'))
    # Preprocess the data (add your preprocessing logic here)
    X = data.values  # Assuming X is your input data
    # Make predictions using the loaded model
    predictions = model.predict(np.expand_dims(X, axis=-1))
    # Return the predictions as JSON (customize the response as needed)
    return {'predictions': predictions.tolist()}

if __name__ == '__main__':
    app.run(debug=True)
