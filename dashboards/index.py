from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from app import app
from apps import statistics_univariate
from dashboards.simple_styling import *

all_tracks = statistics_univariate.df['Track Title'].sort_values()
all_tracks_lists = all_tracks.str.lower().str.split(' ')


sidebar_style = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '16rem',
    'padding': '2rem 1rem',
    'font': main_font,
    'background-color': 'lightgrey'
}
content_style = {
    'margin-left': '16rem',
    'margin-right': '2rem',
    'padding': '1rem 1rem',
    'font': main_font,
    'background-color': 'lightgrey'
}
homelink_style = {
    'font-weight': 'bold',
    'width': '5rem',
    'margin-top': '1rem',
    'margin-bottom': '1rem',
    'font': main_font,
    'color': 'black'
}
dropdownmenu_style = {
    'margin-top': '1rem',
    'margin-bottom': '1rem',
    'font': main_font,
    'color': 'grey'
}
dropdownmenuitem_style = {
    'background': 'white',
    'color': 'black',
    'font': main_font
}

sidebar = html.Div(
    [
        html.H2('Music', className='display-3'),
        html.Hr(),
        html.P(
            'Dataset visualizations', className='lead'
        ),
        dcc.Input(id='searchbar',
                  type='text',
                  placeholder='Search track',
                  list='suggested-tracks',
                  style=dropdownmenu_style,
                  value='',
                  persistence=False),
        html.Datalist(id='suggested-tracks', children=[html.Option(value='')]),
        dbc.Nav(
            [
                dbc.NavLink('Home', href='/', id='nav-home', active='exact',
                            style=homelink_style),
                dbc.DropdownMenu(
                    [
                        dbc.DropdownMenuItem(
                            'Univariate', id='nav-stat-univariate',
                            href='/statistics/univariate', active='exact',
                            style=dropdownmenuitem_style
                        ),
                        dbc.DropdownMenuItem(
                            'Bivariate', id='nav-stat-bivariate',
                            href='/statistics/bivariate', active='exact',
                            style=dropdownmenuitem_style
                        ),
                        dbc.DropdownMenuItem(
                            'Group comparisons', id='nav-stat-groupcomparisons',
                            href='/statistics/group-comparisons', active='exact',
                            style=dropdownmenuitem_style
                        ),
                        dbc.DropdownMenuItem(
                            'Artists', id='nav-stat-artists',
                            href='/statistics/artists', active='exact',
                            style=dropdownmenuitem_style
                        ),
                    ],
                    label='Statistics',
                    style=dropdownmenu_style,
                    in_navbar=True,
                    align_end=True
                ),
                dbc.DropdownMenu(
                    [
                        dbc.DropdownMenuItem(
                            'Rekordbox', id='nav-qual-rekordbox',
                            href='/data-quality/rekordbox', active='exact',
                            style=dropdownmenuitem_style
                        ),
                        dbc.DropdownMenuItem(
                            'Spotify and Youtube', id='nav-qual-spotifyyoutube',
                            href='/data-quality/spotify-youtube', active='exact',
                            style=dropdownmenuitem_style
                        ),
                        dbc.DropdownMenuItem(
                            'Waves', id='nav-qual-waves',
                            href='/data-quality/waves', active='exact',
                            style=dropdownmenuitem_style
                        ),
                    ],
                    label='Data Quality',
                    style=dropdownmenu_style
                ),
                dbc.DropdownMenu(
                    [
                        dbc.DropdownMenuItem(
                            'Overview', id='nav-updt-overview',
                            href='/updates/overview', active='exact',
                            style=dropdownmenuitem_style
                        ),
                        dbc.DropdownMenuItem(
                            'Trends', id='nav-updt-trends',
                            href='/updates/trends', active='exact',
                            style=dropdownmenuitem_style
                        ),
                        dbc.DropdownMenuItem(
                            'Comparisons', id='nav-updt-comparisons',
                            href='/updates/comparisons', active='exact',
                            style=dropdownmenuitem_style
                        ),
                    ],
                    label='Updates',
                    style=dropdownmenu_style
                ),
                dbc.DropdownMenu(
                    [
                        dbc.DropdownMenuItem(
                            'Feature importance', id='nav-modl-featureimportance',
                            href='/model-training/feature-importance', active='exact',
                            style=dropdownmenuitem_style
                        ),
                        dbc.DropdownMenuItem(
                            'Model comparisons', id='nav-modl-comparisons',
                            href='/model-training/comparisons', active='exact',
                            style=dropdownmenuitem_style
                        ),
                        dbc.DropdownMenuItem(
                            'Individual track investigation', id='nav-modl-indiv',
                            href='/model-training/individual-tracks', active='exact',
                            style=dropdownmenuitem_style
                        ),
                    ],
                    label='Model training',
                    style=dropdownmenu_style
                )
            ],
            vertical=True,
            pills=True
        )
    ],
    style=sidebar_style
)

content = html.Div(id='page-content', children=[], style=content_style)

app.layout = html.Div(
    [
        dcc.Location(id='url'),
        sidebar,
        content
    ],
    style={
        'background-color': 'lightgrey'}
)


@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def render_page_content(pathname):

    if pathname == '/':  # monitoring (last update, n songs, which dashboards are ready)
        return []
    elif pathname == '/statistics/univariate':
        return statistics_univariate.layout
    elif pathname == '/statistics/bivariate':
        return []
    elif pathname == '/statistics/group-comparisons':
        return []
    elif pathname == '/statistics/artists':
        return []
    elif pathname == '/data-quality/rekordbox':
        return []
    elif pathname == '/data-quality/spotify-youtube':
        return []
    elif pathname == '/data-quality/waves':
        return []
    elif pathname == '/updates/overview':
        return []
    elif pathname == '/updates/trends':
        return []
    elif pathname == '/updates/comparisons':
        return []
    elif pathname == '/model-training/feature-importance':
        return []
    elif pathname == '/model-training/comparisons':
        return []
    elif pathname == 'model-training/individual-tracks':
        return []
    else:
        return [
            html.H1('404: Not found', className='text-danger'),
            html.Hr(),
            html.P(f'The pathname {pathname} was not recognized')
        ]

@app.callback(
    Output('suggested-tracks', 'children'),
    Input('searchbar', 'value'),
)
def search(track_name):
    if len(track_name) < 3:
        raise PreventUpdate
    track_name = track_name.lower()
    suggestions_list = all_tracks.loc[all_tracks.str.lower().apply(
        lambda x: all([tn in x for tn in track_name.split(' ')]))]

    return [html.Option(value=sug) for sug in suggestions_list]

if __name__ == '__main__':
    app.run_server(debug=True)
