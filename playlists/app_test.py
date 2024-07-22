from base.helpers import open_html_in_browser
from typing import List

import plotly.io as pio
import dash
from dash import html, dcc, ctx, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import pickle
import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

pio._base_renderers.open_html_in_browser = open_html_in_browser
pio.renderers.default = "browser"

model = []
df = pd.DataFrame({
    'id': ['a', 'b', 'c', 'd'],
    'energy': [.9, .7, .8, .9],
    'danceability': [.5, .7, .6, .5],
    'Hardlopen': ['in', 'out', 'out', np.nan],
    'Hardlopen_predict': [np.nan, np.nan, np.nan, np.nan]
})
data = df.to_dict('records')

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    # html.Button('train-model', id='train-model'),
    html.Button('add-data', id='add-data'),
    dcc.Store(
        'stored-model',
        data=pickle.dumps(model)
    ),
    dcc.Store('stored-data', data=data),
    dash_table.DataTable(
        id='df-playlist',
        data=data,
        columns=[{'id': c, 'name': c} for c in df.columns],
        page_size=10
    )
])

@app.callback(
    Output('stored-data', 'data'),
    Input('add-data', 'n_clicks'),
    Input('stored-model', 'data'),
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def add_data(n_clicks, model, data):
    df = pd.DataFrame(data)
    df.loc[df['Hardlopen'].isna(), 'Hardlopen'] = 'in'
    return df.to_dict('records')

@app.callback(
    Output('stored-model', 'data'),
    Input('stored-data', 'data'),
    prevent_initial_call=True
)
def train_model(data):
    df = pd.DataFrame(data)
    df_train = df.loc[~df['Hardlopen'].isna()]
    X = df_train[['energy', 'danceability']]
    y = df_train['Hardlopen']
    model = LGBMClassifier()
    model.fit(X, y)

    joblib.dump(model, 'D:\\Data Science\\Python zelfstudie\\Music\\files\\playlists\\app\\models\\model1.sav')
    return pickle.dumps(model)

@app.callback(
    Output('stored-data', 'data'),
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def predict_data(model, data):
    df = pd.DataFrame(data)
    predict_mask = df.loc[df['Hardlopen'].isna()]
    X = df.loc[predict_mask, ['energy', 'danceability']]
    model = pickle.loads(model)
    if not isinstance(model, List):
        predictions = model.predict(X)
        df.loc[predict_mask, 'Hardlopen_predict'] = predictions

    return df.to_dict('records')

if __name__ == '__main__':
    app.run_server(debug=True, port=55003)