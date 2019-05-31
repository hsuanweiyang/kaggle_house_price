import numpy as np
from sys import argv
import sys
import pandas as pd
from xgboost import XGBRegressor
import os
import multiprocessing
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def generate_fold(input_data, fold_num=3):
    n_sample = input_data.shape[0]
    permutation = np.random.permutation(n_sample)
    shuffled_data = input_data.iloc[permutation, :]
    train_fold = []
    valid_fold = []
    one_fold_num = n_sample//fold_num
    for i in range(fold_num):
        valid_idx = range(i*one_fold_num, (i+1)*one_fold_num)
        train_idx = np.setdiff1d(range(n_sample), valid_idx)
        train = shuffled_data.iloc[train_idx, :]
        valid = shuffled_data.iloc[valid_idx, :]
        train_fold.append(train)
        valid_fold.append(valid)
    return train_fold, valid_fold


def train_xg(train_data, valid_data, parameters):
    feature = train_data.iloc[:, 1:]
    label = train_data.iloc[:, 0]
    valid_feature = valid_data.iloc[:, 1:]
    valid_label = valid_data.iloc[:, 0]
    reg = XGBRegressor(learning_rate=parameters['learning_rate'], min_child_weight=parameters['min_child_weight'],
                       max_depth=parameters['max_depth'], n_estimators=parameters['n_estimator'])
    reg.fit(feature, label)
    valid_result = reg.predict(valid_feature)
    return rmsle(valid_result, valid_label)


def rmsle(y_pred, y_true):
    return np.sqrt(np.mean(np.square(np.log1p(y_pred)-np.log1p(y_true))))


def train_with_crossvalidation(parameters):
    train_fold, valid_fold = generate_fold(train_data, cv_fold_num)
    average_rmsle = 0
    for each_fold in range(cv_fold_num):
        average_rmsle += (train_xg(train_fold[each_fold], valid_fold[each_fold], parameters)) / cv_fold_num
    return average_rmsle, parameters


def single_process_parameter_tuning(train_data, cv_fold):
    learning_rate_range = np.arange(0.01, 0.21, 0.03)
    min_child_weight_range = np.arange(1, 4, 1)
    max_depth_range = np.arange(3, 11, 1)
    n_estimator_range = np.arange(40, 500, 20)
    best_rmsle = 100
    i = 0
    for learning_rate in learning_rate_range:
        for min_child_weight in min_child_weight_range:
            for max_depth in max_depth_range:
                for n_estimator in n_estimator_range:
                    parameters = {'learning_rate': learning_rate, 'min_child_weight': min_child_weight,
                                  'max_depth': max_depth, 'n_estimator': n_estimator}
                    current_rmsle, _ = train_with_crossvalidation(train_data, [cv_fold, parameters])
                    if current_rmsle < best_rmsle:
                        best_rmsle = current_rmsle
                        best_parameters = parameters
                    i += 1
                    sys.stdout.write('Tuning Progress:\t{0} %\t rmsle:\t{1}\r'.format(round((100*i)/3381, 2), best_rmsle))
                    sys.stdout.flush()

    return best_rmsle, best_parameters


def multi_process_parameter_tuning():
    parameter_pair = generate_parameter_pair()
    multiprocess = multiprocessing.Pool(3)
    results = multiprocess.map(train_with_crossvalidation, parameter_pair)
    multiprocess.close()
    multiprocess.join()
    best_rmsle = 100
    best_parameters = {'learning_rate': -1, 'min_child_weight': 0, 'max_depth': 100}
    for result in results:
        if result[0] < best_rmsle:
            best_rmsle = result[0]
        elif result[0] == best_rmsle:
            if result[1]['learning_rate'] > best_parameters['learning_rate']:
                best_parameters['learning_rate'] = result[1]['learning_rate']
            if result[1]['min_child_weight'] > best_parameters['min_child_weight']:
                best_parameters['min_child_weight'] = result[1]['min_child_weight']
            if result[1]['max_depth'] < best_parameters['max_depth']:
                best_parameters['max_depth'] = result[1]['max_depth']
            if result[1]['n_estimator'] < best_parameters['n_estimator']:
                best_parameters['n_estimator'] = result[1]['n_estimator']
    return best_rmsle, best_parameters


def generate_parameter_pair():
    learning_rate_range = np.arange(0.01, 0.21, 0.03)
    min_child_weight_range = np.arange(1, 4, 1)
    max_depth_range = np.arange(3, 11, 1)
    n_estimator_range = np.arange(40, 500, 20)
    pair = []
    for learning_rate in learning_rate_range:
        for min_child_weight in min_child_weight_range:
            for max_depth in max_depth_range:
                for n_estimator in n_estimator_range:
                    pair.append({'learning_rate': learning_rate, 'min_child_weight': min_child_weight,
                                  'max_depth': max_depth, 'n_estimator': n_estimator})
    return pair



if __name__ == '__main__':
    input_train_file = argv[1]
    input_test_file = argv[2]
    cv_fold_num = int(argv[3])

    train_data = pd.read_pickle(input_train_file)
    test_data = pd.read_pickle(input_test_file)
    multi_process_parameter_tuning()

