from dashboards.app import app
from dash import html, dcc, dash_table, callback
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.figure_factory as ff

from music.database import MySQLdb
from dashboards.simple_styling import *

import numpy as np
import pandas as pd
import re
from scipy import stats
from sklearn.neighbors import KernelDensity


def convert_milliseconds_to_minutes_seconds(milliseconds):
    seconds, milliseconds = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)

    return f'{str(int(minutes)).zfill(2)}:{str(int(seconds)).zfill(2)}'


def create_theta_labels(df):
    tmp = df.groupby('Key')['id'].count().to_frame('count').reset_index()
    tmp['Tone'] = tmp['Key'].apply(lambda x: int(re.sub('[^0-9]', '', x)))
    tmp['Level'] = tmp['Key'].apply(lambda x: re.sub('[^A-Z]', '', x))
    tmp = tmp.sort_values(by=['Tone', 'Level']).drop(['Tone', 'Level'], axis=1).reset_index(drop=True)
    theta = list(tmp['count'])
    labels = list(tmp['Key'])

    return theta, labels


def create_x_vals(series):
    N = series.size
    smin = series.min()
    smax = series.max()
    x = np.linspace(
        smin,
        smax,
        N)

    return x

def create_distline(series):
    x = create_x_vals(series)

    bw = series.std() * .25
    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(np.array(series).reshape(-1, 1))
    y = np.exp(kde.score_samples(x.reshape(-1, 1)))

    return x, y


def create_norm_line(series, avg, std):
    x = create_x_vals(series)
    y = stats.norm.pdf(x, avg, std)

    return x, y


# pio.renderers.default = 'browser'
db_config = {'host': 'localhost',
             'username': 'root',
             'password': 'aHp4mCm6!',
             'db_name': 'music'}

db = MySQLdb(db_config)
ids = db.load_table('tracks_my_id')
df = db.load_table('tracks_my_rekordbox')
last_version = max([c for c in ids.columns if c.startswith('version_')])
df = df.merge(ids[['id', 'File Name', last_version]], how='left', on='id')
df = df.loc[df[last_version] == 1]
df['File Type'] = df['File Name'].apply(lambda x: '.' + x.split('.')[-1].lower())

features_hist = ['BPM', 'Time']
features_bar = ['Rating', 'Bitrate', 'Genre', 'Track Kind', 'File Type', 'Beat sync']
genres = ['Afro_Disco',
          'Balearic', 'Cosmic', 'Disco', 'Italo_Disco', 'Nu_Disco', 'Acid_House',
          'Deep_House', 'House', 'Indie', 'Techno', 'Nostalgia', 'Old_Deep_House']

layout = html.Div(
    [
        html.Div(
            [
                dbc.Row([
                    dbc.Card(
                        [
                            dbc.Row(
                                [
                                    dbc.Col([
                                        'Feature'
                                    ],
                                        width=3),
                                    dbc.Col([
                                        'Distribution'
                                    ],
                                        width=9)
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    dcc.Dropdown(id='feature_dropdown_hist',
                                                                 options=features_hist,
                                                                 value='BPM'),
                                                    html.Hr(),
                                                    dash_table.DataTable(id='histogram-table',
                                                                         row_selectable='multi',
                                                                         selected_rows=[],
                                                                         style_cell_conditional=
                                                                         [
                                                                             {
                                                                                 'if': {'column_id': 'Metric'},
                                                                                 'textAlign': 'left'
                                                                             }
                                                                         ],
                                                                         style_data={
                                                                             'color': 'black',
                                                                             'backgroundColor': 'white',
                                                                             'font-family': main_font
                                                                         },
                                                                         style_header={
                                                                             'backgroundColor': 'rgb(210, 210, 210)',
                                                                             'color': 'black',
                                                                             'fontWeight': 'bold',
                                                                             'font-family': main_font
                                                                         },
                                                                         style_data_conditional=
                                                                         [
                                                                             {'if': {
                                                                                 'filter_query': '{Value} = <.05 || {Value} = <.01',
                                                                                 'column_id': 'Value'},
                                                                              'backgroundColor': '#ff7851'},
                                                                             {'if': {
                                                                                 'filter_query': '{Value} >= .05 && {Metric} = `KS normality pval`',
                                                                                 'column_id': 'Value',
                                                                                 },
                                                                              'backgroundColor': '#56cc9d'}
                                                                         ]
                                                                         )
                                                ]
                                            )
                                        ],
                                        width=3
                                    ),
                                    dbc.Col(
                                        [
                                            dcc.Graph(id='histogram')
                                        ],
                                        width=9
                                    )
                                ]
                            )
                        ],
                        style=card_style
                    )
                ]),
                dbc.Row([
                    dbc.Card(
                        [
                            dbc.Row(
                                [
                                    dbc.Col([
                                        'Count'
                                    ],
                                        width=6),
                                    dbc.Col([
                                        'Camelot keys'
                                    ],
                                        width=6)
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dcc.Dropdown(id='feature_dropdown_bar',
                                                         options=features_bar,
                                                         value='Rating'),
                                            dcc.Graph(id='horizontal-bar')
                                        ],
                                        width=6
                                    ),
                                    dbc.Col(
                                        [
                                            dcc.Graph(id='bar-polar')
                                        ],
                                        width=6
                                    )
                                ]
                            )
                        ],
                        style=card_style
                    )
                ])
            ]
        ),
    ]
)


@callback(
    Output('histogram', 'figure'),
    Output('histogram-table', 'data'),
    Output('histogram-table', 'columns'),
    [Input('feature_dropdown_hist', 'value'),
     Input('histogram-table', 'selected_rows')]
)
def histogram(feature, selected_rows, data=df[features_hist + ['rb_duration', 'Track Title']]):
    feature_backend = 'rb_duration' if feature == 'Time' else feature
    series = data[feature_backend]
    Q1 = series.quantile(.25)
    Q3 = series.quantile(.75)
    IQR = Q3 - Q1
    outliers = (series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)
    data['Outlier'] = outliers
    n_outliers = outliers.sum()

    metrics = {'Mean': series.mean(),
               'Median': series.median(),
               'Mode': series.mode().iloc[0],
               'Quantiles (0.25%, 0.75%)': [Q1, Q3],
               'Standard deviation': series.std(),
               'Skewness': series.skew(),
               'Kurtosis': series.kurtosis(),
               'N outliers': n_outliers}
    normality = stats.kstest((series - metrics['Mean']) / metrics['Standard deviation'], 'norm')
    metrics.update({'KS normality metric': normality.statistic,
                    'KS normality pval': normality.pvalue})
    metrics = {k: (round(v, 2) if k != 'Quantiles (0.25%, 0.75%)' else v) for k, v in metrics.items()}

    if metrics['KS normality pval'] < .01:
        metrics['KS normality pval'] = '<.01'
    elif metrics['KS normality pval'] < .05:
        metrics['KS normality pval'] = '<.05'

    selected_metrics = [list(metrics.keys())[i] for i in selected_rows]

    hover_dict = {feature: True}
    if feature == 'Time':
        hover_dict.update({feature_backend: False})
    if 'N outliers' in selected_metrics:
        hover_dict.update({'Outlier': True})
    fig = px.histogram(data,
                       x=feature_backend,
                       marginal='rug',
                       color=main_color*data.shape[0],
                       color_discrete_map='identity',
                       hover_data=hover_dict,
                       hover_name='Track Title',
                       histnorm='probability density')
    opacity = .8
    if feature == 'Time':
        ticks = np.linspace(series.min(), series.max(), 8)
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=ticks,
                ticktext=[convert_milliseconds_to_minutes_seconds(t) for t in ticks],
                title=feature
            ),
            xaxis2=dict(
                tickmode='array',
                tickvals=ticks
            )
        )

    fig.update_layout(showlegend=False,
                      plot_bgcolor='rgba(0, 0, 0, 0)',
                      paper_bgcolor='rgba(0, 0, 0, 0)',
                      xaxis=dict(
                          gridcolor=grid_color),
                      xaxis2=dict(
                          gridcolor=grid_color),
                      yaxis=dict(
                          gridcolor=grid_color,
                          zeroline=False,
                          showticklabels=True,
                          title=None),
                      autosize=False,
                      margin=dict(
                          b=0,
                          t=30,
                          l=0,
                          r=0
                      ))

    for i in [sr for sr in selected_rows if sr <= 2]:
        y_pos = .75 + i * .05
        fig.add_vline(x=metrics[list(metrics.keys())[i]],
                      line_color='darkgrey',
                      row=1,
                      line_dash='dash')
        fig.add_annotation(x=metrics[list(metrics.keys())[i]],
                           yref='paper', y=y_pos,
                           text=list(metrics.keys())[i], showarrow=False)

    if 'Quantiles (0.25%, 0.75%)' in selected_metrics:
        y_pos = .7
        for Q, Qname in zip([Q1, Q3], [.25, .75]):
            fig.add_vline(x=Q,
                          line_color='lightgrey',
                          row=1,
                          line_dash='dash')
            fig.add_annotation(x=Q,
                               yref='paper', y=y_pos,
                               text=str(Qname) + '%', showarrow=False)

    if 'Standard deviation' in selected_metrics:
        fig.add_shape(x0=metrics['Mean'] - metrics['Standard deviation'], x1=metrics['Mean'] + metrics['Standard deviation'],
                      yref='paper', y0=0, y1=.057,
                      fillcolor='darkgrey', opacity=.3, line_width=0)
        fig.add_vline(x=metrics['Mean'], yref='paper', y0=0, y1=.057, row=1)
        fig.add_annotation(x=metrics['Mean'] - metrics['Standard deviation'],
                           yref='paper', y=.007,
                           text='-1'+u'\u03C3', showarrow=False)
        fig.add_annotation(x=metrics['Mean'] + metrics['Standard deviation'],
                           yref='paper', y=.007,
                           text='+1'+u'\u03C3', showarrow=False)

    if any([s_k in selected_metrics for s_k in ['Skewness', 'Kurtosis']]):
        dist_x, dist_y = create_distline(series)
        fig.add_trace(go.Scatter(x=dist_x, y=dist_y, mode='lines',
                                 marker=dict(color=main_color[0],
                                             line=dict(width=3)),
                                 fill='tozeroy', fillcolor='rgba(255,207,153,.35)'))
        opacity = .15

    if 'N outliers' in selected_metrics:
        fig.add_shape(x0=series.min(),
                      x1=Q1 - 1.5 * IQR,
                      yref='paper', y0=0, y1=np.inf,
                      fillcolor=pos_neg_colors[1], opacity=.2, line_width=0)
        fig.add_annotation(x=Q1 - 1.5 * IQR,
                           yref='paper', y=.5,
                           text='Q1 - 1.5 * IQR',
                           showarrow=False,
                           align='left')
        fig.add_shape(x0=Q3 + 1.5 * IQR,
                      x1=series.max(),
                      yref='paper', y0=0, y1=np.inf,
                      fillcolor=pos_neg_colors[1], opacity=.2,
                      line_width=0)
        fig.add_annotation(x=Q3 + 1.5 * IQR,
                           yref='paper', y=.5,
                           text='Q3 + 1.5 * IQR',
                           showarrow=False,
                           align='right')

    if any([s_k in selected_metrics for s_k in ['KS normality metric', 'KS normality pval']]):
        norm_x, norm_y = create_norm_line(series, metrics['Mean'], metrics['Standard deviation'])
        fig.add_trace(go.Scatter(x=norm_x, y=norm_y, mode='lines',
                                 marker=dict(color=main_color[0],
                                             line=dict(width=3)),
                                 fill='tozeroy', fillcolor='rgba(0,0,0,.25)'))

    if feature == 'Time':
        for m in ['Mean', 'Median', 'Mode', 'Standard deviation']:
            metrics[m] = convert_milliseconds_to_minutes_seconds(metrics[m])

        metrics['Quantiles (0.25%, 0.75%)'] = [convert_milliseconds_to_minutes_seconds(q)
                                               for q in metrics['Quantiles (0.25%, 0.75%)']]

    fig.update_traces(marker={'opacity': opacity})
    metrics['Quantiles (0.25%, 0.75%)'] = str(metrics['Quantiles (0.25%, 0.75%)'])
    cols = ['Metric', 'Value']
    metrics_records = pd.DataFrame([[k, v] for k, v in metrics.items()], columns=cols).to_dict('records')
    metrics_columns = [{'id': c, 'name': c} for c in cols]

    return fig, metrics_records, metrics_columns


# @callback(
#     Output('historgram-data', 'data'),
#     Input('feature-dropdown-hist')
# )
# def get_metrics_hist(feature, data=df):
#


@callback(
    Output('horizontal-bar', 'figure'),
    Input('feature_dropdown_bar', 'value')
)
def bar(feature, data=df):
    if feature == 'Track Kind':
        feature = 'track_kind'
    if feature in ['Rating', 'Bitrate', 'track_kind', 'File Type']:
        grouped_data = data.groupby(feature)['id'].count().sort_values(ascending=True)
        x = list(grouped_data.index)
        y = list(grouped_data.values)
    elif feature == 'Beat sync':
        x = ['Offbeat', 'Onbeat']
        n_offbeat = data['Offbeat'].sum()
        y = [n_offbeat, data.shape[0] - n_offbeat]
    else:
        series_genres = pd.Series([data[g].sum() for g in genres], index=genres).sort_values()
        x = [g.replace('_', ' ') for g in series_genres.index]
        y = list(series_genres.values)

    fig = go.Figure([go.Bar(x=y, y=x, orientation='h')])
    fig.update_traces(
        marker={'color': main_color[0],
                'opacity': .8},
        hovertemplate=f'{feature} ' + ': %{y} <br>Count: %{x}<extra></extra>')
    fig.update_yaxes(title=feature.capitalize().replace('_', ' '))
    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',
                      paper_bgcolor='rgba(0, 0, 0, 0)',
                      xaxis=dict(gridcolor=grid_color,
                                 title='Count'),
                      yaxis=dict(gridcolor=grid_color),
                      margin=dict(
                          b=0,
                          t=0,
                          l=0,
                          r=0
                      )
                      )

    return fig


@callback(
    Output('bar-polar', 'figure'),
    [Input('searchbar', 'value')]
)
def bar_polar(searched_track, data=df):
    theta, labels = create_theta_labels(data)
    radial_tick_min = int(max(theta) / 10)
    radial_tick_max = int(max(theta) / 10) * 10
    radial_tick_avg = int((radial_tick_max - radial_tick_min) / (np.log(radial_tick_max) - np.log(radial_tick_min)))
    radial_ticks = [radial_tick_min, radial_tick_avg, radial_tick_max]
    fig = px.bar_polar(r=theta, theta=labels, start_angle=67.5,
                       color_discrete_sequence=main_color * 24)
    fig.update_layout(
        paper_bgcolor='rgba(0, 0, 0, 0)',
        showlegend=False,
        polar=dict(
            radialaxis=dict(
                type='log',
                tickangle=0,
                tickvals=radial_ticks
            ),
            angularaxis=dict(
                thetaunit='degrees',
                direction='clockwise',
                tickmode='array'
            )
        ),
        margin=dict(
            b=15,
            t=15,
            l=0,
            r=0
        )
    )
    fig.update_polars(
        bgcolor='rgba(0,48,103,0)',
        angularaxis_gridcolor=grid_color,
        radialaxis_gridcolor=grid_color
    )
    fig.update_traces(
        marker={'opacity': .8},
        hovertemplate='Key: %{theta} <br>Count: %{r}'
    )

    return fig
