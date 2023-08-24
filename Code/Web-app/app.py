from flask import Flask, render_template, request, redirect
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)


# Load the pre-trained CNN model


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        Rpm = float(request.form['Rpm'])  # Use lowercase 'Rpm' to match the HTML form
        Feed = float(request.form['Feed'])  # Use lowercase 'Feed' to match the HTML form
        Depth = float(request.form['Depth'])  # Use lowercase 'Depth' to match the HTML form

        # Create a DataFrame with user input
        data = {'Rpm': [Rpm], 'Feed': [Feed], 'Depth': [Depth]}
        input_df = pd.DataFrame(data)

        # Save the DataFrame to an Excel file
        input_df.to_excel('user_input.xlsx', index=False)
        datum = pd.read_excel('user_input.xlsx')
        # Preprocess the data
        # You would need to preprocess the data similar to how you preprocessed your training data
        # scaler = MinMaxScaler()
        # Experiment details
        model = load_model('C:/Users/moury/AIA/FSM-INT-2023/Code/cnn_tuner.h5')

        exp = pd.read_excel('C:/Users/moury/AIA/FSM-INT-2023/Code/Data/Experiment Summary.xlsx')
        dataframes = []  # To store the imported data from each file

        for i in range(1, 61):
            file = f"C:/Users/moury/AIA/FSM-INT-2023/Code/Data/{i}.xlsx"
            df = pd.read_excel(file)  # Use pd.read_excel() for Excel files
            df = df.dropna(axis='columns', how='all')
            df = df.dropna(axis='rows', how='all')
            df.columns = ['Time', 'X', 'Y', 'Z']
            df = df.iloc[1:]  # Exclude the original header row from the data
            df['Time'] = pd.to_datetime(df['Time'],
                                        unit='s').dt.time  # Convert 'Time' column to standard datetime format.

            # Adding experiment details from exp dataframe based on experiment number

            experiment_number = i
            experiment_row = exp[exp['Experiment'] == experiment_number]
            if not experiment_row.empty:
                for column in experiment_row.columns[1:]:
                    df[column] = experiment_row[column].values[0]

            dataframes.append(df)

        # converting to numeric values and calculating the magnitude of vibrational data.

        for i in range(len(dataframes)):
            dataframes[i]['X'] = pd.to_numeric(dataframes[i]['X'], errors='coerce')
            dataframes[i]['Y'] = pd.to_numeric(dataframes[i]['Y'], errors='coerce')
            dataframes[i]['Z'] = pd.to_numeric(dataframes[i]['Z'], errors='coerce')
            dataframes[i]['Magnitude'] = np.sqrt(
                dataframes[i]['X'] ** 2 + dataframes[i]['Y'] ** 2 + dataframes[i]['Z'] ** 2)

        # concatenating all the vibration data together from 60 files.
        merged_df = pd.concat(dataframes, ignore_index=True)
        merged_df['Time'] = [t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6 for t in merged_df['Time']]
        # concatenating all the vibration data together from 60 files.
        merged_df = pd.concat(dataframes, ignore_index=True)
        merged_df['Time'] = [t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6 for t in merged_df['Time']]

        # Split the data into features (X) and target variable (y)
        columns_to_drop = ['Ra', 'X', 'Y', 'Z', 'Time',
                           'Magnitude']  # dropping X, Y, X, Time, Magnitude as they had no impact on the result when
        # tested.
        X = merged_df.drop(columns_to_drop, axis=1)
        y = merged_df['Ra']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

        # Scale the features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(datum)

        # Make predictions using the pre-trained CNN model
        predictions = model.predict(np.expand_dims(X_test_scaled, axis=-1))  # Replace with your preprocessed data

        # predicted_value = predictions[0]  # Assuming you're predicting a single value

        return render_template('result.html', predicted_value=predictions[0])

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
