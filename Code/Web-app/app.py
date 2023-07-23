from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import io

app = Dash(__name__)

app.layout = html.Div([
    html.H1(children='Dashboard', style={'textAlign': 'center'}),
    dcc.Dropdown(id='dropdown-selection'),
    dcc.Graph(id='graph-content'),
    dcc.Upload(
        id='upload-data',
        children=html.Button('Upload Excel File')
    ),
])


@app.callback(
    Output('dropdown-selection', 'options'),
    Output('graph-content', 'figure'),
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename')]
)
def update_options_and_graph(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename:
                # Read the file using pandas
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            elif 'xls' in filename:
                # Read the file using pandas
                df = pd.read_excel(io.BytesIO(decoded))
            else:
                return [], "Unsupported file format"
        except Exception as e:
            print(e)
            return [], "Error processing file"

        # Update dropdown options with column names from the uploaded file
        options = [{'label': col, 'value': col} for col in df.columns]

        # Select the first column as the default option in the dropdown
        selected_value = df.columns[0]

        # Your plotly graph update code here using 'df' and 'selected_value'
        fig = px.line(df, x=selected_value, y='pop')

        return options, fig

    else:
        # Fallback options and graph when no file is uploaded
        df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')
        options = [{'label': col, 'value': col} for col in df.columns]
        selected_value = df.columns[0]
        fig = px.line(df, x=selected_value, y='pop')

        return options, fig


if __name__ == '__main__':
    app.run_server(debug=True)
