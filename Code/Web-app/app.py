import dash
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import os
import base64
import io  # Correct import for BytesIO

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Upload Excel Files", style={'textAlign': 'center'}),
    dcc.Upload(
        id='upload-data',
        children=html.Button('Click here to Vibration data files'),
        style={
            'width': '50%',
            'height': '100px',
            'lineHeight': '100px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': 'auto'
        },
        multiple=True  # Allow multiple files to be uploaded
    ),
    dcc.Upload(
        id='upload-exp',
        children=html.Button('Click here to Upload experiment data File'),
        style={
            'width': '50%',
            'height': '100px',
            'lineHeight': '100px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': 'auto'
        }
    ),
    html.Div(id='output-data')
])


@app.callback(
    Output('output-data', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents_list, filename_list):
    if contents_list is None or len(contents_list) == 0:
        raise PreventUpdate

    all_dfs = []  # To store all the DataFrames from the uploaded files

    for content, filename in zip(contents_list, filename_list):
        if content is not None:
            content_type, content_string = content.split(',')
            try:
                if 'csv' in filename:
                    # Read the file using pandas
                    df = pd.read_csv(io.StringIO(base64.b64decode(content_string).decode('utf-8')))
                elif 'xls' in filename or 'xlsx' in filename:
                    # Read the file using pandas
                    df = pd.read_excel(io.BytesIO(base64.b64decode(content_string)))
                else:
                    return "Unsupported file format"
            except Exception as e:
                print(e)
                return "Error processing file"

            # Process the DataFrame (if needed)
            df = df.dropna(axis='columns', how='all')
            df = df.dropna(axis='rows', how='all')
            df.columns = ['Time', 'X', 'Y', 'Z']
            df = df.iloc[1:]
            df['Time'] = pd.to_datetime(df['Time'], unit='s').dt.time

            # Your additional DataFrame processing code here
            # ...

            all_dfs.append(df)


'''
    # Now you have all the DataFrames in the `all_dfs` list
    # You can store them in the `dataframes` list or process them further

    # For example, you can print the first few rows of each DataFrame:


    output_children = []
    for idx, df in enumerate(all_dfs):
        output_children.append(html.Div([
            html.H3(f"Uploaded File ({idx + 1}): {filename_list[idx]}"),
            html.P(df.head().to_html())
        ]))

    return output_children
'''

if __name__ == '__main__':
    app.run_server(debug=True)
