import pandas as pd

SUBMISSION_FOLDER = 'submissions'
DATA_FOLDER = 'data'


def get_type(data, threshold):
    total_actions = dict(data.action_type.value_counts())
    data['type'] = data.apply(
        lambda row: row['action_type'] if total_actions[row['action_type']] >= threshold else row['combined_shot_type'],
        axis=1)
    return


def get_time_remaining(data, threshold):
    data['time_remaining'] = data.apply(lambda row: row['minutes_remaining'] * 60 + row['seconds_remaining'], axis=1)
    anomaly = 14
    data['last_moment'] = data.apply(lambda row: row['time_remaining'] < threshold or row['time_remaining'] == anomaly,
                                     axis=1)
    return


def get_away(data):
    data['away'] = data.matchup.str.contains('@')
    return


def get_season(data):
    data['season'] = data.apply(lambda row: int(row['season'].split('-')[0]), axis=1)
    return


def get_month(data):
    data['month'] = data.apply(lambda row: int(row['game_date'].split('-')[1]), axis=1)
    return


def fix_shot_distance(data):
    data['shot_distance'] = data.apply(lambda row: 28 if row['shot_distance'] > 28 else row['shot_distance'], axis=1)
    return


def load_data(path, small=False, part=10):
    data = pd.read_csv(DATA_FOLDER + '/' + path)

    get_type(data, 100)
    get_time_remaining(data, 3)
    get_away(data)
    get_season(data)
    get_month(data)
    fix_shot_distance(data)

    features = [
        'shot_distance',
        'last_moment',
        'away',
    ]

    binary_features = [
        'period',
        'shot_zone_area',
        'shot_zone_basic',
        'shot_zone_range',
        'opponent',
        'season',
        'action_type',
        'combined_shot_type'
    ]

    dummies_counter = {}
    for feature in binary_features:
        dummies_counter[feature] = len(data[feature].unique())

    data = pd.get_dummies(data, columns=binary_features)

    for col in data.columns.tolist():
        for feature in binary_features:
            if col.startswith(feature) and col != feature:
                features.append(col)

    target = 'shot_made_flag'
    id_col = 'shot_id'

    new_columns = list(features)
    new_columns.reverse()
    new_columns.append(id_col)
    new_columns.reverse()
    new_columns.append(target)
    data = data[new_columns]
    train_data = data[~data[target].isnull()]
    if small:
        train_data = train_data[::part]
    test_data = data[data[target].isnull()]
    test_data.drop(target, axis=1, inplace=True)

    return train_data, test_data, features, target


def load_meta(path_train, path_test):
    train = pd.read_csv(DATA_FOLDER + '/' + path_train)
    test = pd.read_csv(DATA_FOLDER + '/' + path_test)
    return train, test


def make_submission(predictions, ids, file_name):
    with open(SUBMISSION_FOLDER + '/' + file_name, 'w') as f:
        f.write('shot_id,shot_made_flag\n')
        for i, prob in zip(ids, predictions):
            f.write(str(i) + ',' + str(prob) + '\n')
    print('Wrote submission to file {}.'.format(file_name))
