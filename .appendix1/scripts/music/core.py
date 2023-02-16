import logging
from joblib import load, dump
import pandas as pd
from music import subprocesses, helpers


def check_rekordbox_data_altered(rekordbox_filename,
                                 app_filename):
    rekordbox_data = pd.read_csv(rekordbox_filename,
                                 sep=None,
                                 header=0,
                                 encoding='utf-16',
                                 engine='python').to_dict('records')
    version = helpers.get_latest_version(file_kind='music', music_dataset='my', dataset_type='app', latest=True)
    my_music_app = load(f'{app_filename}_{version}.sav')
    rb_filenames = [d['File Name'] for d in rekordbox_data]
    app_filenames = [d['File Name'] for d in my_music_app]

    all_app_files_in_rb = all([fn in rb_filenames for fn in app_filenames])
    all_rb_files_in_app = all([fn in app_filenames for fn in rb_filenames])

    rekordbox_data_altered = not all([all_app_files_in_rb, all_rb_files_in_app])

    return rekordbox_data_altered


def get_feature_categories(feature_categories_filename):
    feature_categories = load(feature_categories_filename)
    features = [v for v in feature_categories.values()]
    model_data_features = ['File Name']
    model_data_features.extend(feature_categories['librosa'])
    model_data_features.extend(feature_categories['chord'])
    model_data_features.extend(['key', 'mode', 'tempo'])
    model_data_features.extend(['sp_danceability', 'sp_energy', 'sp_valence'])

    return feature_categories, features, model_data_features


feature_selection = True
retrain_baseline = True
retrain_models = True

log_filename = 'logs.txt'
feature_categories_filename = 'feature_categories_my.sav'
my_raw_data_filename = 'music_my_raw'
my_model_data_filename = 'music_my_model'
my_app_data_filename = 'music_my_app'
full_model_data_filename = 'music_full_model'
scaled_model_data_filename = 'music_scaled_model'
rekordbox_filename = 'rekordbox_file.txt'
tracks_dir = './tracks/my'
scaler_filename = 'standard_scaler'
targets = ['danceability', 'energy', 'valence']
feature_categories, features, model_data_features = get_feature_categories(feature_categories_filename)

logging.basicConfig(filename=log_filename,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

rekordbox_data_altered = check_rekordbox_data_altered(rekordbox_filename=rekordbox_filename,
                                                      app_filename=my_app_data_filename)

if rekordbox_data_altered:

    my_dataset = subprocesses.DatasetMy(feature_categories=feature_categories,
                                        features=features,
                                        model_data_features=model_data_features,
                                        my_raw_data_filename=my_raw_data_filename,
                                        my_app_data_filename=my_app_data_filename,
                                        my_model_data_filename=my_model_data_filename,
                                        rekordbox_filename=rekordbox_filename,
                                        tracks_dir=tracks_dir,
                                        logging=logging)
    my_dataset.create()

    full_model_dataset = subprocesses.DatasetFull(full_model_data_filename=full_model_data_filename)
    full_model_dataset.create()
    ranges_got_wider = full_model_dataset.check_ranges(feature_categories)
    full_model_dataset.scale(scaled_model_data_filename=scaled_model_data_filename,
                             scaler_filename=scaler_filename)

    if ranges_got_wider:

        if feature_selection:

            if retrain_baseline:
                lasso = subprocesses.LassoFeatureSelection(scaled_model_data_filename)
                lasso.execute()

                razor_lasso = subprocesses.RazorFeatureSelection('lasso')
                razor_lasso.execute(lasso.data)  # lasso.data

            boruta = subprocesses.BorutaFeatureSelection(scaled_model_data_filename)
            boruta.execute()

            razor_boruta = subprocesses.RazorFeatureSelection('boruta')
            razor_boruta.execute(boruta.data)

    if retrain_models:

        linear_feature_selections = ['lasso', 'razor_lasso']
        non_linear_feature_selections = ['boruta', 'razor_boruta']
        train_datasets = ['full', 'my', 'random']
        train_test_match_case = {'full': ['full', 'my', 'random'],
                                 'my': ['my'],
                                 'random': ['random']}
        test_sets = {td: helpers.create_test_set(td) for td in train_datasets}
        counter = 0
        for train_dataset in train_datasets:
            for test_dataset in train_test_match_case[train_dataset]:
                test_tracks = test_sets[test_dataset]

                for target in targets:
                    counter += 1
                    print(f'{train_dataset} ; {test_dataset}; {target} ; {counter} / 15')

                    for feature_selection in linear_feature_selections:

                        linear = subprocesses.TrainLinear(train_dataset=train_dataset,
                                                          test_dataset=test_dataset,
                                                          feature_selection=feature_selection,
                                                          test_tracks=test_tracks,
                                                          target=target)
                        linear.execute()
                    for feature_selection in non_linear_feature_selections:
                        lgbm = subprocesses.TrainLGBM(train_dataset=train_dataset,
                                                      test_dataset=test_dataset,
                                                      feature_selection=feature_selection,
                                                      test_tracks=test_tracks,
                                                      target=target)
                        lgbm.execute()
                        nn = subprocesses.TrainNeuralNetwork(train_dataset=train_dataset,
                                                             test_dataset=test_dataset,
                                                             feature_selection=feature_selection,
                                                             test_tracks=test_tracks,
                                                             target=target)
                        nn.execute()

        model_kinds = ['linear', 'lgbm', 'neuralnetwork']
        comparison = subprocesses.CompareModels(train_datasets=train_datasets,
                                                test_sets=test_sets,
                                                train_test_match_case=train_test_match_case,
                                                targets=targets,
                                                model_kinds=model_kinds)
        comparison.create_results()
