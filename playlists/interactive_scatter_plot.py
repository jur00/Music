import pickle

from config import Config
from base.helpers import open_html_in_browser, adaptive_linspace

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import fbeta_score, recall_score, precision_score, balanced_accuracy_score, make_scorer

from joblib import load, dump
from itertools import product
import time
from datetime import datetime
import re
import os
from typing import List

pio._base_renderers.open_html_in_browser = open_html_in_browser
pio.renderers.default = "browser"

def get_last_model(playlist):
    location = Config.playlists_dir + f'app\\models\\'
    models = list(sorted([m for m in os.listdir(location) if m.startswith(playlist)]))
    if len(models) > 0:
        return load(f'{location}{models[-1]}')
    else:
        return []


class SearchSpace:

    def __init__(self, playlist, results_map, first_model):
        self.playlist = playlist
        self.results_map = results_map
        self.first_model = first_model

        self.hyperparameter_inits = self._set_hyperparameter_inits()
        self.search_space_options = self._set_search_space_options()
        self.hyperparameters = list(self.hyperparameter_inits.keys())

        self.grid_search_n = None
        self.df_hyperparameter_values = None
        self.search_space = None
        self.init_values = None

    @staticmethod
    def _set_hyperparameter_inits():
        return dict(
            n_estimators=100,
            max_depth=3,
            num_leaves=5,
            min_child_samples=5,
            reg_alpha=.1,
            reg_lambda=1
        )

    @staticmethod
    def _regularization_logspace(n=12, base=10):
        return np.round((np.logspace(0, 1, n, base=base) - 1) / (base - 1), 4)

    def _set_search_space_options(self):
        return dict(
            n_estimators=np.arange(50, 1001, 25),
            max_depth=np.arange(3, 32, 1),
            num_leaves=np.arange(3, 32, 1),
            min_child_samples=np.arange(3, 32, 1),
            reg_alpha=self._regularization_logspace(),
            reg_lambda=self._regularization_logspace()
        )

    def _get_previous_results(self):
        if self.first_model:
            return [], 0
        else:
            previous_grid_searches = list(sorted(os.listdir(self.results_map)))
            previous_grid_search = previous_grid_searches[-1]
            previous_grid_search_n = int(previous_grid_search.split('.')[0])
            return (
                [load(f'{self.results_map}{pgs}')['best_param'] for pgs in previous_grid_searches],
                previous_grid_search_n + 1
            )

    def _get_hyperparameter_values_df(self, previous_grid_searches):
        return pd.DataFrame(
            [self.hyperparameter_inits] + [gs.best_params_ for gs in previous_grid_searches]
        )

    def _get_previous_hyperparameter(self):
        if self.first_model:
            return 'reg_lambda'
        else:
            previous_hyperparameter_values = self.df_hyperparameter_values.iloc[-1]
            return previous_hyperparameter_values.loc[~previous_hyperparameter_values.isna()].index[0]

    def _get_current_hyperparameter(self, previous_hyperparameter):
        if previous_hyperparameter == self.hyperparameters[-1]:
            return self.hyperparameters[0]
        else:
            return self.hyperparameters[self.hyperparameters.index(previous_hyperparameter) + 1]

    def _get_previous_hyperparameter_values(self):
        previous_hyperparameter_values = dict()
        for hyperparameter in self.hyperparameters:
            hyperparameter_values = self.df_hyperparameter_values[hyperparameter]
            previous_value = hyperparameter_values.loc[~hyperparameter_values.isna()].to_list()[-1]
            previous_hyperparameter_values[hyperparameter] = previous_value
        return previous_hyperparameter_values

    @staticmethod
    def _find_nearest_values(arr, val):
        idx_nearest = np.argmin(np.abs(arr - val))
        idx_min = max([0, idx_nearest - 1])
        idx_max = min([len(arr), idx_nearest + 1])

        return arr[idx_min:idx_max + 1]

    def _prepare_search_space(self, previous_hyperparameter_values, current_hyperparameter):
        previous_value = previous_hyperparameter_values[current_hyperparameter]
        search_space_options = self.search_space_options[current_hyperparameter]
        return {current_hyperparameter: self._find_nearest_values(search_space_options, previous_value)}

    @staticmethod
    def _prepare_init_values(previous_hyperparameter_values, current_hyperparameter):
        return {k: v for k, v in previous_hyperparameter_values.items() if k != current_hyperparameter}

    def run(self):
        previous_grid_searches, self.grid_search_n = self._get_previous_results()
        self.df_hyperparameter_values = self._get_hyperparameter_values_df(previous_grid_searches)

        previous_hyperparameter = self._get_previous_hyperparameter()
        current_hyperparameter = self._get_current_hyperparameter(previous_hyperparameter)
        previous_hyperparameter_values = self._get_previous_hyperparameter_values()

        self.search_space = self._prepare_search_space(previous_hyperparameter_values, current_hyperparameter)
        self.init_values = self._prepare_init_values(previous_hyperparameter_values, current_hyperparameter)


class ModelTrain:

    def __init__(self, df, features, playlist):
        self.df = df
        self.features = features
        self.playlist = playlist

        # file settings
        self.model_path = Config.playlists_models_dir + f'{playlist}.sav'
        self.results_map = Config.playlists_models_dir + f'{playlist}_results\\'
        self.first_model = os.path.exists(self.results_map)

        self.ssp = SearchSpace(playlist, self.results_map, self.first_model)
        self.ssp.run()

        self.results_path = f'{self.results_map}{self.ssp.grid_search_n}.sav'

        self.X = None
        self.y = None
        self.y_bool = None
        self.grid_searches = None
        self.gs_n_features = None
        self.best_n_features = None
        self.X_optim = None
        self.best_param = None
        self.init_values = None
        self.scoring = None
        self.model = None
        self.scores = None
        self.results = None

    @staticmethod
    def _datetime_ext():
        return re.sub(r'[^0-9]', '', str(datetime.now()).split('.')[:-1][0])

    def _prepare_data(self):
        train_mask = ~self.df[self.playlist].isna()
        self.X = self.df.loc[train_mask, self.features]
        self.y = self.df.loc[train_mask, self.playlist]
        self.y_bool = self.y.replace({'in': 1, 'out': 0})

    def _grid_search(self, X):
        model = LGBMClassifier(**self.ssp.init_values)
        param_grid = self.ssp.search_space
        scoring = make_scorer(fbeta_score, beta=.5)
        grid_search = GridSearchCV(
            estimator=model, param_grid=param_grid, cv=5, scoring=scoring, return_train_score=True
        )
        grid_search.fit(X, self.y_bool)
        return grid_search

    def _grid_search_n_features(self):
        self.grid_searches = dict()
        self.gs_n_features = dict()
        for n_features in [2, 3, 4]:
            X = self.X.iloc[:, :n_features]
            grid_search = self._grid_search(X)
            self.grid_searches[n_features] = grid_search
            grid_search_results = {'score': grid_search.best_score_, 'params': grid_search.best_params_}
            self.gs_n_features[n_features] = {k: v for k, v in grid_search_results.items()}

    def _get_best_n_features(self):
        r = {k: v['score'] - ((k-2) * .01) for k, v in self.gs_n_features.items()}
        self.best_n_features = max(r, key=r.get)

    def _prepare_x_optim_and_init_values(self):
        self.X_optim = self.X.iloc[:, :self.best_n_features]
        self.init_values = self.ssp.init_values
        self.best_param = self.gs_n_features[self.best_n_features]['params']
        self.init_values.update(self.best_param)

    def _prepare_scoring(self):
        self.scoring = {
            'balanced_accuracy': make_scorer(balanced_accuracy_score),
            'fbeta': make_scorer(fbeta_score, beta=.5),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score)
        }

    def _fit_model(self):
        self.scores = cross_validate(
            LGBMClassifier(**self.init_values),
            self.X_optim,
            self.y_bool,
            cv=5,
            scoring=self.scoring,
            return_train_score=True
        )
        self.model = LGBMClassifier(**self.init_values)
        self.model.fit(self.X_optim, self.y)

    def _prepare_results(self):
        feature_importance = {f: i for f, i in zip(self.model.feature_name_, self.model.feature_importances_)}
        other_features = list(set(self.features) - set(self.model.feature_name_))
        feature_importance.update({of: 0 for of in other_features})

        scores = {k: np.mean(v) for k, v in self.scores.items() if k.startswith('test_')}

        self.results = {
            'feature_importance': feature_importance,
            'scores': scores,
            'best_param': mt.best_param
        }

    def _save(self):
        dump(self.model, self.model_path)

        if self.first_model:
            os.makedirs(self.results_map)

        dump(self.results, self.results_path)

    def run(self):
        self._prepare_data()
        self._grid_search_n_features()
        self._get_best_n_features()
        self._prepare_x_optim_and_init_values()
        self._prepare_scoring()
        self._fit_model()
        self._prepare_results()
        self._save()

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
            marker_color='navy',
            text=self.s_class_balance,
            textposition='auto'
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
        self.track_name = None
        self.data_table = None
        self.add_disable = None
        self.exclude_disable = None

    def _get_file_name(self):
        self.file_name = self.clickData['points'][0]['customdata'][7]
        
    def _get_track_name(self):
        artist = self.clickData['points'][0]['customdata'][1]
        track = self.clickData['points'][0]['customdata'][0]
        self.track_name = f'{artist} - {track}'

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
            self._get_track_name()
            self._add_remove_reset()
            self._save_df()

        self._filter_data_table()
        self.toggle_disabled_buttons()
            


playlist = 'Hardlopen'
features = ['energy', 'tempo', 'danceability', 'valence']

df = load(Config.df_app)
model = get_last_model(playlist)

tm = TrainedModel(df, model, playlist, features)

mt = ModelTrain(df, features, playlist)
mt.run()
