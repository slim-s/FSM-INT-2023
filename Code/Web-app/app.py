import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_file
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the ML model
model = load_model('cnn.h5')

# Store the uploaded experiment and vibration files
experiment_df = None
vibration_dataframes = []


@app.route('/')
def index():
    return render_template('index.html', predictions=None)


@app.route('/upload_experiment', methods=['POST'])
def upload_experiment():
    global experiment_df
    experiment_file = request.files['experiment_file']
    if experiment_file:
        # Process the experiment data
        experiment_df = pd.read_excel(experiment_file)
        return redirect(url_for('index'))


@app.route('/upload_vibration', methods=['POST'])
def upload_vibration():
    global vibration_dataframes
    vibration_files = request.files.getlist('vibration_files')
    if vibration_files:
        for file in vibration_files:
            # Process the vibration data
            df = pd.read_excel(file)
            df = df.dropna(axis='columns', how='all')
            df = df.dropna(axis='rows', how='all')
            df.columns = ['Time', 'X', 'Y', 'Z']
            df = df.iloc[1:]
            df['Time'] = pd.to_datetime(df['Time'], unit='s').dt.time
            df['X'] = pd.to_numeric(df['X'], errors='coerce')
            df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
            df['Z'] = pd.to_numeric(df['Z'], errors='coerce')
            df['Magnitude'] = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)

            # Add experiment details from experiment_df based on experiment number
            experiment_number = int(df['Experiment'].iloc[0])  # Assuming 'Experiment' is a column in df
            if experiment_df is not None:
                experiment_row = experiment_df[experiment_df['Experiment'] == experiment_number]
                if not experiment_row.empty:
                    for column in experiment_row.columns[1:]:
                        df[column] = experiment_row[column].values[0]

            vibration_dataframes.append(df)
            print(f"Processed: {file.filename}")

        return redirect(url_for('index'))


@app.route('/predict', methods=['GET'])
def predict():
    global vibration_dataframes
    if not vibration_dataframes:
        return redirect(url_for('index'))

    # Concatenate all vibration dataframes into merged_df
    merged_df = pd.concat(vibration_dataframes, ignore_index=True)
    merged_df['Time'] = [(t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6) for t in merged_df['Time']]

    # Prepare the data for prediction
    scaler = MinMaxScaler()
    X_test_scaled = scaler.fit_transform(merged_df.drop(['Time', 'Magnitude'], axis=1))

    # Make predictions using the loaded model
    predictions = model.predict(np.expand_dims(X_test_scaled, axis=-1))
    merged_df['Predictions'] = predictions

    # Save the predictions to an Excel file
    predictions_file = 'predictions.xlsx'
    merged_df.to_excel(predictions_file, index=False)

    # Clear vibration_dataframes for next prediction
    vibration_dataframes = []

    return send_file(predictions_file, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
