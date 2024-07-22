from config import Config
from base.helpers import open_html_in_browser
from playlists.interactive_scatter_plot import (
    InteractiveScatterPlot, FeatureImportancePlot, DfUpdate, DfLoading, ClassBalancePlot,
    ModelTrain, TrainedModel, get_last_model
)

import pandas as pd
import numpy as np
from joblib import load, dump
import pickle
import json
from typing import List

import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pygame import mixer

import dash
from dash import html, dcc, ctx, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

pio._base_renderers.open_html_in_browser = open_html_in_browser
pio.renderers.default = "browser"

n_seconds_skip = 30

class Audio:

    def __init__(self):
        self.pos = 0

    def play(self, file_name):
        self.pos = 0
        track_file = Config.my_tracks_dir + file_name
        mixer.init()
        mixer.music.load(track_file)
        mixer.music.play()

    def skip(self, seconds=n_seconds_skip):
        mixer.music.set_pos(self.pos)
        self.pos += seconds

    def stop(self):
        mixer.music.stop()

audio = Audio()
track_name = ''

playlist = 'Hardlopen'

df_liked = pd.read_csv(Config.playlists_dir + 'tracks_liked.csv')
df_my = pd.DataFrame(load(Config.df_my_music_no_wave_features_path))

df = load(Config.df_app)
model = get_last_model(playlist)

features = ['energy', 'tempo', 'danceability', 'valence']
playlists = ['Hardlopen', 'Borrel', 'Etentje', 'After']
customdata_cols = ['name', 'artists', 'energy', 'danceability', 'valence', 'tempo', 'popularity', 'File Name']
train_model_threshold = 55

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout
app.layout = dbc.Container([
    html.H1("Playlist Picker", className="my-4"),
    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Card([
                    dbc.CardHeader("Playlist"),
                    dbc.CardBody(
                        dcc.Dropdown(
                            id='playlist-dropdown',
                            options=[{'label': pl, 'value': pl} for pl in playlists],
                            value='Hardlopen'
                        )
                    )
                ], className="mb-4")
            ]),
            dbc.Row([
                dbc.Card([
                    dbc.CardHeader("Heatmap"),
                    dbc.CardBody([
                        html.Div([
                            "Kind",
                            dcc.RadioItems(
                                id='heatmap-kind-radio',
                                options=['bool', 'float'],
                                value='float'
                            ),
                        ]),
                        html.Div([
                            "Resolution",
                            dcc.RadioItems(
                                id='heatmap-resolution-radio',
                                options=['low', 'high'],
                                value='high'
                            )
                        ])
                    ])
                ], className="mb-4")
            ])

        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Features"),
                dbc.CardBody([
                    html.Div([
                        "Left x",
                        dcc.Dropdown(
                            id='left-x-dropdown',
                            options=[{'label': f, 'value': f} for f in features],
                            value='energy'
                        )
                    ]),
                    html.Div([
                        "Left y",
                        dcc.Dropdown(
                            id='left-y-dropdown',
                            options=[{'label': f, 'value': f} for f in features],
                            value='tempo'
                        ),
                    ]),
                    html.Div([
                        "Right x",
                        dcc.Dropdown(
                            id='right-x-dropdown',
                            options=[{'label': f, 'value': f} for f in features],
                            value='valence'
                        )
                    ]),
                    html.Div([
                        "Right y",
                        dcc.Dropdown(
                            id='right-y-dropdown',
                            options=[{'label': f, 'value': f} for f in features],
                            value='danceability'
                        ),
                    ])
                ])
            ], className="mb-4")
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Feature Importance"),
                dbc.CardBody(
                    dcc.Graph(id='feature-importance-plot')
                )
            ], className="mb-4")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('Model Performance'),
                dbc.CardBody('')
            ])
        ], width=1),
        dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Class Balance'),
                    dbc.CardBody(
                        dcc.Graph(id='class-balance-plot')
                    )
            ])
        ], width=2)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Scatter Plot"),
                dbc.CardBody(
                    dcc.Graph(id='scatter-plot')
                )
            ], className='mb-4')
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.Div(id='audio-output')),
                dbc.CardBody([
                    html.Button('Stop', id='audio-stop', n_clicks=0),
                    html.Button(f'Skip {n_seconds_skip} seconds', id='audio-skip', n_clicks=0),
                    html.Button('Add to playlist', id='add-track', n_clicks=0, disabled=True),
                    html.Button('Exclude from playlist', id='exclude-track', n_clicks=0, disabled=True),
                    html.Button('Reset playlist', id='reset-playlist', n_clicks=0),
                    dcc.Store('stored-data', data=df.to_dict('records'))
                ])
            ], className='mb-4')
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('data'),
                dbc.CardBody(dash_table.DataTable(
                    id='df-playlist',
                    data=df.to_dict('records'),
                    columns=[{'id': c, 'name': c} for c in ['name', 'artists', playlist]],
                    page_size=10
                ))
            ], className='mb-4')
        ], width=6)
    ]),

], fluid=True)

# Update scatter plot based on dropdown selections
@app.callback(
    Output('scatter-plot', 'figure'),
    Output('feature-importance-plot', 'figure'),
    Output('class-balance-plot', 'figure'),
    Input('playlist-dropdown', 'value'),
    Input('left-x-dropdown', 'value'),
    Input('left-y-dropdown', 'value'),
    Input('right-x-dropdown', 'value'),
    Input('right-y-dropdown', 'value'),
    Input('heatmap-kind-radio', 'value'),
    Input('heatmap-resolution-radio', 'value'),
    Input('scatter-plot', 'clickData'),
    Input('stored-data', 'data')
)
def update_scatter_plot(
        playlist, left_x, left_y, right_x, right_y, heatmap_kind,
        heatmap_resolution, clickData, updated_data
):
    if len(np.unique([left_x, left_y, right_x, right_y])) < 4:
        raise ValueError('all axis inputs must be different')

    df = pd.DataFrame(updated_data)
    n_tracks_decided = (~df[playlist].isna()).sum()
    train_model = n_tracks_decided >= train_model_threshold

    if train_model:
        mt = ModelTrain(df, features, playlist)
        mt.run()
        model = mt.model

        tm = TrainedModel(df, model, playlist, features)
        tm.run()

        df = tm.df
        s_feature_importance = tm.s_feature_importance
    else:
        model = []
        s_feature_importance = pd.Series({f: 0 for f in features})

    isp = InteractiveScatterPlot(
        df,
        model,
        left_x,
        left_y,
        right_x,
        right_y,
        playlist,
        heatmap_kind,
        heatmap_resolution,
        features,
        customdata_cols,
        clickData,
        train_model
    )
    isp.run()

    fip = FeatureImportancePlot(s_feature_importance, features, train_model_threshold)
    fip.run()

    cbp = ClassBalancePlot(df, playlist)
    cbp.run()

    return isp.fig, fip.fig, cbp.fig

# Play audio when a marker is clicked
@app.callback(
    Output('audio-output', 'children'),
    Input('scatter-plot', 'clickData'),
    Input('audio-skip', 'n_clicks'),
    Input('audio-stop', 'n_clicks')
)
def audio_player(clickData, btn_skip, btn_stop):
    global audio
    global track_name

    button_id = ctx.triggered_id if not None else 'No clicks yet'

    if button_id == 'scatter-plot':
        track_name = clickData['points'][0]['customdata'][0]
        file_name = clickData['points'][0]['customdata'][7]
        audio = Audio()
        audio.play(file_name)
        return f"Playing audio: {track_name}"
    elif button_id == 'audio-skip':
        audio.skip()
        return f"Playing audio: {track_name}"
    elif button_id == 'audio-stop':
        audio.stop()
        return 'Click a marker to play audio.'
    else:
        return "Click a marker to play audio."

@app.callback(
    Output('df-playlist', 'data'),
    Output('stored-data', 'data'),
    Output('add-track', 'disabled'),
    Output('exclude-track', 'disabled'),
    Input('playlist-dropdown', 'value'),
    Input('add-track', 'n_clicks'),
    Input('exclude-track', 'n_clicks'),
    Input('reset-playlist', 'n_clicks'),
    Input('scatter-plot', 'clickData'),
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def add_remove_tracks(playlist, add, remove, reset, clickData, current_data):
    button_id = ctx.triggered_id if not None else 'No clicks yet'

    df = DfLoading(playlist).load()

    udf = DfUpdate(df, playlist, button_id, clickData)
    udf.run()

    return udf.data_table, udf.df.to_dict('records'), udf.add_disable, udf.exclude_disable

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=55003)
