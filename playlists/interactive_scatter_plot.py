import pickle

from config import Config
from base.helpers import open_html_in_browser, adaptive_linspace

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from joblib import load, dump
from itertools import product
import time
from datetime import datetime
import re
import os

pio._base_renderers.open_html_in_browser = open_html_in_browser
pio.renderers.default = "browser"

def get_last_model(playlist):
    location = Config.playlists_dir + f'app\\models\\'
    models = list(sorted([m for m in os.listdir(location) if m.startswith(playlist)]))
    if len(models) > 0:
        return load(f'{location}{models[-1]}')
    else:
        return []

class ModelTrain:

    def __init__(self, df, features, playlist):
        self.df = df
        self.features = features
        self.playlist = playlist

        self.save_filename = Config.playlists_dir + f'app\\models\\{playlist}.sav'

        self.X = None
        self.y = None
        self.model = None

    @staticmethod
    def _datetime_ext():
        return re.sub(r'[^0-9]', '', str(datetime.now()).split('.')[:-1][0])

    def _prepare_data(self):
        train_mask = ~self.df[self.playlist].isna()
        self.X = self.df.loc[train_mask, self.features]
        self.y = self.df.loc[train_mask, self.playlist]

    def _fit_model(self):
        self.model = LGBMClassifier()
        self.model.fit(self.X, self.y)

    def _save_model(self):
        dump(self.model, self.save_filename)

    def run(self):
        self._prepare_data()
        self._fit_model()
        self._save_model()

class TrainedModel:

    def __init__(self, df, model, playlist, features):
        self.df = df
        self.model = model
        self.playlist = playlist
        self.features = features

        self.s_feature_importance = None
        self.starting_features = None

    def _get_feature_importance(self):
        self.s_feature_importance = pd.Series(self.model.feature_importances_, index=self.features)

    def _get_starting_features(self):
        self.starting_features = list(
            self.s_feature_importance
            .sort_values(ascending=False)
            .index[:2]
        )

    def _predict_in_out(self):
        predict_mask = self.df[self.playlist].isna()
        X = self.df.loc[predict_mask, self.features]
        self.df.loc[predict_mask, f'{self.playlist}_predict'] = self.model.predict(X)

    def run(self):
        self._get_feature_importance()
        self._get_starting_features()
        self._predict_in_out()


class DecisionMeshgrid:

    def __init__(self, df, sign_f_x, sign_f_y, redu_f_x, redu_f_y, features, model, resolution='high'):

        self.df = df
        self.sign_f_x = sign_f_x
        self.sign_f_y = sign_f_y
        self.redu_f_x = redu_f_x
        self.redu_f_y = redu_f_y
        self.features = features
        self.model = model
        self.resolution = resolution

        self.mins_maxs = None
        self.sign_len = None
        self.redu_len = None
        self.grid_vars = None
        self.df_grid = None
        self.df_predictions = None
        self.df_predict_grid = None
        self.df_meshgrid_bool = None
        self.df_meshgrid_float = None
        self.df_bound_coordinates = None

    def _get_mins_maxs(self):
        self.mins_maxs = {
            f: {'min': self.df[f].min(),
                'max': self.df[f].max()}
            for f in self.features
        }

    def _set_sign_redu_len(self):
        if self.resolution == 'high':
            self.sign_len = 60
            self.redu_len = 5
        else:
            self.sign_len = 30
            self.redu_len = 4

    def _create_grid_vars(self):
        self.grid_vars = {}
        grid_var_signs = {
            var: np.linspace(self.mins_maxs[var]['min'], self.mins_maxs[var]['max'], self.sign_len)
            for var in [self.sign_f_x, self.sign_f_y]
        }
        grid_var_redus = {
            var: adaptive_linspace(self.df[var], self.redu_len)
            for var in [self.redu_f_x, self.redu_f_y]
        }

        self.grid_vars.update(grid_var_signs)
        self.grid_vars.update(grid_var_redus)

    def _create_grid_df(self):
        self.df_grid = pd.DataFrame(list(product(*[self.grid_vars[f] for f in self.features])), columns=self.features)

    def _create_predictions_df(self):
        self.df_predictions = pd.DataFrame(self.model.predict_proba(self.df_grid), columns=self.model.classes_)

    @staticmethod
    def __choose_class(proba):
        if proba >= .5:
            return 'in'
        else:
            return 'out'

    def _create_predict_grid_df(self):
        self.df_predict_grid = (
            pd.concat([self.df_grid, self.df_predictions], axis=1)
            .groupby([self.sign_f_x, self.sign_f_y])
            ['in']
            .mean()
            .to_frame('probability')
            .reset_index()
            .assign(prediction=lambda xdf: xdf['probability'].apply(self.__choose_class))
            .assign(prediction_int=lambda xdf: xdf['prediction'].replace({'in': 1, 'out': 0}))
        )

    def _create_meshgrid_dfs(self):
        self.df_meshgrid_bool = pd.pivot(self.df_predict_grid, self.sign_f_x, self.sign_f_y, 'prediction_int')
        self.df_meshgrid_float = pd.pivot(self.df_predict_grid, self.sign_f_x, self.sign_f_y, 'probability')

    def __get_decision_boundaries(self, axis):
        symbol_plot = 'line-ns' if axis == 0 else 'line-ew'

        df_bounds = self.df_meshgrid_bool.diff(axis=axis).stack()
        df_bound_coordinates_part = (
            df_bounds[df_bounds != 0]
                .to_frame('ones')
                .reset_index()
                .drop(columns='ones')
                .assign(symbol_plot=symbol_plot)
        )
        return df_bound_coordinates_part

    def _create_decision_bounds_df(self):
        self.df_bound_coordinates = (
            pd.concat([self.__get_decision_boundaries(axis) for axis in [0, 1]])
            .round(3)
            .reset_index(drop=True)
        )

    def run(self):
        self._get_mins_maxs()
        self._set_sign_redu_len()
        self._create_grid_vars()
        self._create_grid_df()
        self._create_predictions_df()
        self._create_predict_grid_df()
        self._create_meshgrid_dfs()
        self._create_decision_bounds_df()


def make_decision_meshgrid(df, model,  left_x, left_y, right_x, right_y, features, resolution):
    sign_f_x = left_x
    sign_f_y = left_y
    redu_f_x = right_x
    redu_f_y = right_y

    dm = DecisionMeshgrid(
        df=df,
        sign_f_x=sign_f_x,
        sign_f_y=sign_f_y,
        redu_f_x=redu_f_x,
        redu_f_y=redu_f_y,
        features=features,
        model=model,
        resolution=resolution
    )
    dm.run()

    return dm.df_meshgrid_bool, dm.df_meshgrid_float, dm.df_bound_coordinates


class ProbabilityHeatmapPlot:

    def __init__(
        self,
        left_x,
        left_y,
        right_x,
        right_y,
        df_meshgrid_bool,
        df_meshgrid_float,
        df_bound_coordinates,
        heatmap_kind
    ):
        self.df_meshgrid_bool = df_meshgrid_bool
        self.df_meshgrid_float = df_meshgrid_float
        self.df_bound_coordinates = df_bound_coordinates
        self.heatmap_kind = heatmap_kind
        self.left_x = left_x
        self.left_y = left_y
        self.right_x = right_x
        self.right_y = right_y
        self.df_meshgrid = self.df_meshgrid_bool if heatmap_kind == 'bool' else self.df_meshgrid_float

        self.fig = None
        self.colorscale = None
        self.min_prob = None
        self.max_prob = None
        self.avg_prob = None

    def _initialize_figure(self):
        self.fig = make_subplots(
            rows=1,
            cols=2,
            horizontal_spacing=.04
        )

    def _plot_boundaries(self):
        if self.heatmap_kind == 'float':
            self.fig.add_scatter(
                mode='markers',
                x=self.df_bound_coordinates[self.left_x],
                y=self.df_bound_coordinates[self.left_y],
                marker=dict(
                    color='grey',
                    symbol=self.df_bound_coordinates['symbol_plot'],
                    line_width=1
                ),
                name='decision boundary',
                row=1,
                col=1
            )

    def _plot_heatmap(self):
        self.fig.add_heatmap(
            z=self.df_meshgrid.to_numpy().transpose(),
            x=self.df_meshgrid.index,
            y=self.df_meshgrid.columns,
            colorscale='RdYlGn',
            colorbar=dict(
                title='background<br>probability',
                # len=.85,
                x=-.035,
                xanchor='right',
                y=0,
                yanchor='bottom',
                ticks='inside'
            ),
            opacity=.15,
            hovertemplate='probability: %{z:.3f}<extra></extra>',
            row=1,
            col=1
        )

    def _adjust_layout(self):
        self.fig.update_xaxes(
            gridcolor='lightgrey'
        )
        self.fig.update_yaxes(
            gridcolor='lightgrey'
        )
        self.fig.update_coloraxes(
            showscale=False
        )
        self.fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)'
        )

    def plot(self, show):
        self._initialize_figure()
        self._plot_boundaries()
        self._plot_heatmap()
        self._adjust_layout()
        if show:
            self.fig.show()


class InteractiveScatterPlot:

    def __init__(
            self,
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
    ):
        self.df = df
        self.model = model
        self.left_x = left_x
        self.left_y = left_y
        self.right_x = right_x
        self.right_y = right_y
        self.playlist = playlist
        self.heatmap_kind = heatmap_kind
        self.heatmap_resolution = heatmap_resolution
        self.features = features
        self.customdata_cols = customdata_cols
        self.clickData = clickData
        self.train_model = train_model

        self.df_in = None
        self.df_out = None
        self.df_undecided = None
        self.df_in_predict = None
        self.df_out_predict = None
        self.n_tracks_decided = None
        self.mf = None
        self.fig = None
        self.grid_dfs = None
        self.grid_colors = None
        self.grid_marker_symbols = None
        self.grid_marker_line_width = None
        self.grid_names = None

    def _prepare_data(self):
        self.df_in = self.df.loc[self.df[self.playlist] == 'in']
        self.df_out = self.df.loc[self.df[self.playlist] == 'out']

    def _create_prediction_dfs(self):
        self.df_in_predict = self.df.loc[self.df[f'{self.playlist}_predict'] == 'in']
        self.df_out_predict = self.df.loc[self.df[f'{self.playlist}_predict'] == 'out']

    def _plot_probability_heatmap(self):
        df_meshgrid_bool, df_meshgrid_float, df_bound_coordinates = make_decision_meshgrid(
            self.df,
            self.model,
            self.left_x,
            self.left_y,
            self.right_x,
            self.right_y,
            self.features,
            self.heatmap_resolution
        )

        php = ProbabilityHeatmapPlot(
            self.left_x,
            self.left_y,
            self.right_x,
            self.right_y,
            df_meshgrid_bool,
            df_meshgrid_float,
            df_bound_coordinates,
            self.heatmap_kind
        )
        php.plot(show=False)

        self.fig = php.fig

    def _create_undecided_df(self):
        self.df_undecided = self.df.loc[self.df[self.playlist].isna()]

    def _initialize_figure(self):
        self.fig = make_subplots(
            rows=1,
            cols=2,
            horizontal_spacing=.04
        )

    def _create_aesthetics(self):
        if self.train_model:
            self.grid_dfs = [self.df_in, self.df_out, self.df_in_predict, self.df_out_predict]
            self.grid_colors = ['green', 'red'] * 2
            self.grid_marker_symbols = ['circle', 'circle', 'x-thin', 'x-thin']
            self.grid_marker_line_width = [0, 0, 2, 2]
            self.grid_names = ['in', 'out', 'in_predicted', 'out_predicted']
        else:
            self.grid_dfs = [self.df_in, self.df_out, self.df_undecided]
            self.grid_colors = ['green', 'red', 'black']
            self.grid_marker_symbols = ['circle', 'circle', 'circle-open']
            self.grid_marker_line_width = [0, 0, 2]
            self.grid_names = ['in', 'out', 'undecided']

    @staticmethod
    def _extend_name(i):
        name_ext = 'left' if i == 0 else 'right'
        return name_ext

    def _plot_scatters(self):
        for i, (x_var, y_var) in enumerate(zip([self.left_x, self.right_x], [self.left_y, self.right_y])):
            name_ext = self._extend_name(i)
            for df_plot, color, symbol, line_width, name in zip(
                    self.grid_dfs, self.grid_colors, self.grid_marker_symbols,
                    self.grid_marker_line_width, self.grid_names
            ):
                self.fig.add_scatter(
                    mode='markers',
                    x=df_plot[x_var],
                    y=df_plot[y_var],
                    marker=dict(
                        color=color,
                        size=14,
                        symbol=symbol,
                        line=dict(
                            color=color,
                            width=line_width
                        )
                    ),
                    opacity=.8,
                    name=f'{name}_{name_ext}',
                    customdata=df_plot[self.customdata_cols],
                    hovertemplate=
                    "<b>%{customdata[0]}</b><br>" +
                    "<i>%{customdata[1]}</i><br><br>" +
                    "energy: %{customdata[2]}<br>" +
                    "danceability: %{customdata[3]}<br>" +
                    "valence: %{customdata[4]}<br>" +
                    "tempo: %{customdata[5]}<br>" +
                    "popularity: %{customdata[6]}<br>" +
                    "file_name: %{customdata[7]}" +
                    "<extra></extra>",
                    row=1,
                    col=i + 1
                )

    def _plot_click(self):
        for i, (x_var, y_var) in enumerate(zip([self.left_x, self.right_x], [self.left_y, self.right_y])):
            if self.clickData:
                self.fig.add_scatter(
                    mode='markers',
                    x=[self.clickData['points'][0]['customdata'][self.customdata_cols.index(x_var)]],
                    y=[self.clickData['points'][0]['customdata'][self.customdata_cols.index(y_var)]],
                    marker=dict(
                        size=10,
                        color='cornflowerblue',
                        # symbol='triangle-right',
                        line=dict(
                            color='cornflowerblue',
                            width=3
                        )
                    ),
                    showlegend=False,
                    row=1,
                    col=i + 1
                )

    def _adjust_layout(self):
        for i, (x_var, y_var) in enumerate(zip([self.left_x, self.right_x], [self.left_y, self.right_y])):
            self.fig.update_xaxes(
                title=x_var,
                gridcolor='lightgray',
                row=1,
                col=i + 1
            )
            self.fig.update_yaxes(
                title=y_var,
                gridcolor='lightgray',
                row=1,
                col=i + 1
            )

        self.fig.update_layout(
            height=700,
            margin=dict(t=5, b=8),
            plot_bgcolor='rgba(0,0,0,0)',
            clickmode='event+select'
        )

    def run(self, show=False):
        self._prepare_data()

        if self.train_model:
            self._create_prediction_dfs()
            self._plot_probability_heatmap()
        else:
            self._create_undecided_df()
            self._initialize_figure()

        self._create_aesthetics()
        self._plot_scatters()
        self._plot_click()
        self._adjust_layout()
        if show:
            self.fig.show()

class FeatureImportancePlot:

    def __init__(self, s_feature_importance, features, train_model_threshold):
        self.s_feature_importance = s_feature_importance
        self.train_model_threshold = train_model_threshold

        self.feature_importances_exist = self.s_feature_importance.sum() > 0

        self.fig = None

    def _initialize_figure(self):
        self.fig = go.Figure()

    def _plot_bars(self):
        self.fig.add_bar(
            x=self.s_feature_importance.index,
            y=self.s_feature_importance,
            marker_color='navy'
        )

    def _plot_annotation(self):
        if not self.feature_importances_exist:
            self.fig.add_annotation(
                text=f'Reach {self.train_model_threshold} decided tracks<br>to train model',
                showarrow=False,
                x=.5,
                y=.5,
                xref='paper',
                yref='paper'
            )

    def _adjust_layout(self):

        if self.feature_importances_exist:
            self.fig.update_yaxes(gridcolor='lightgrey')
            self.fig.update_xaxes(categoryorder='total descending')
        else:
            self.fig.update_yaxes(showticklabels=False)
            self.fig.update_xaxes(categoryorder='category ascending')

        self.fig.update_layout(
            height=240,
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            margin=dict(t=2, b=2, r=2, l=2)
        )

    def run(self):
        self._initialize_figure()
        self._plot_bars()
        self._plot_annotation()
        self._adjust_layout()

class ClassBalancePlot:

    def __init__(self, df, playlist):
        self.s_class_balance = pd.Series(
            {cl: (df[playlist] == cl).sum() for cl in ['in', 'out']}
        )

        self.fig = None

    def _initialize_figure(self):
        self.fig = go.Figure()

    def _plot_bars(self):
        self.fig.add_bar(
            x=self.s_class_balance.index,
            y=self.s_class_balance,
            marker_color='navy'
        )

    def _adjust_layout(self):
        self.fig.update_xaxes(
            categoryorder='category ascending'
        )
        self.fig.update_yaxes(
            gridcolor='lightgrey'
        )
        self.fig.update_layout(
            title=f'N decided: {self.s_class_balance.sum()}',
            height=240,
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            margin=dict(t=30, b=2, r=2, l=2)
        )

    def run(self):
        self._initialize_figure()
        self._plot_bars()
        self._adjust_layout()

class DfLoading:

    def __init__(self, playlist=None):
        self.playlist = playlist

        self.df = None

    @staticmethod
    def _load_df(max_retries=10, delay=.5):
        for attempt in range(max_retries):
            try:
                df = load(Config.df_app)
                return df
            except EOFError:
                print(f"EOFError encountered. Retrying in {delay} second(s)... ({attempt + 1}/{max_retries})")
                time.sleep(delay)
        raise EOFError(f"Failed to load the file after {max_retries} attempts due to EOFError")

    @staticmethod
    def assign_empty_playlist_column(df, playlist):
        df = df.assign(**{playlist: np.nan, f'{playlist}_predict': np.nan})
        return df

    def _add_non_existing_playlist_to_df(self):
        if self.playlist not in self.df.columns:
            self.df = self.assign_empty_playlist_column(self.df, self.playlist)

    def load(self):
        self.df = self._load_df()
        self._add_non_existing_playlist_to_df()
        return self.df

class DfUpdate:
    
    def __init__(self, df, playlist, button_id, clickData):
        self.df = df
        self.playlist = playlist
        self.button_id = button_id
        self.clickData = clickData

        self.file_name = None
        self.data_table = None
        self.add_disable = None
        self.exclude_disable = None

    def _get_file_name(self):
        self.file_name = self.clickData['points'][0]['customdata'][7]
        
    def _add_remove_reset(self):
        if self.button_id == 'add-track':
            self.df.loc[self.df['File Name'] == self.file_name, self.playlist] = 'in'
        elif self.button_id == 'exclude-track':
            self.df.loc[self.df['File Name'] == self.file_name, self.playlist] = 'out'
        elif self.button_id == 'reset-playlist':
            self.df = DfLoading().assign_empty_playlist_column(self.df, self.playlist)
    
    def _save_df(self):
        dump(self.df, Config.df_app)
    
    def _filter_data_table(self):
        self.data_table = self.df.loc[~self.df[self.playlist].isna(), :].to_dict('records')
    
    def toggle_disabled_buttons(self):
        if self.clickData:
            track_not_decided = self.df.loc[self.df['File Name'] == self.file_name, self.playlist].isna().iat[0]
            track_added = self.df.loc[self.df['File Name'] == self.file_name, self.playlist].iat[0] == 'in'
            track_excluded = self.df.loc[self.df['File Name'] == self.file_name, self.playlist].iat[0] == 'out'
            if track_not_decided:
                self.add_disable = False
                self.exclude_disable = False
            elif track_added:
                self.add_disable = True
                self.exclude_disable = False
            elif track_excluded:
                self.add_disable = False
                self.exclude_disable = True
        else:
            self.add_disable = True
            self.exclude_disable = True
    
    def run(self):
        if self.clickData:
            self._get_file_name()
            self._add_remove_reset()
            self._save_df()

        self._filter_data_table()
        self.toggle_disabled_buttons()
            

left_x = 'energy'
left_y = 'danceability'
right_x = 'valence'
right_y = 'tempo'
playlist = 'Hardlopen'
heatmap_kind = 'float'
heatmap_resolution = 'high'

customdata_cols = ['name', 'artists', 'energy', 'danceability', 'valence', 'tempo', 'popularity', 'File Name']
features = ['energy', 'danceability', 'valence', 'tempo']
train_model_threshold = 44


#
# isp = InteractiveScatterPlot(
#     df,
#     left_x,
#     left_y,
#     right_x,
#     right_y,
#     playlist,
#     heatmap_kind,
#     heatmap_resolution,
#     features,
#     customdata_cols,
#     train_model_threshold
# )
#
# isp.run(show=True)

