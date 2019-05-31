
import pandas as pd
import numpy as np
from sys import argv


def modify_data(df):
    # Handle Missing Data
    df['MasVnrType'] = df['MasVnrType'].fillna('MING')
    df['Electrical'] = df['Electrical'].fillna('SBrkr')

    for cat_col in ['Alley', 'GarageType', 'Fence', 'MiscFeature', 'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2',
                    'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'BsmtExposure',
                    'ExterQual', 'ExterCond', 'HeatingQC', 'CentralAir', 'KitchenQual']:
        df[cat_col] = df[cat_col].fillna('None')

    # Numerical Features
    numerical_features_idx = ['LotArea', 'LotFrontage', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
                              'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                              'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
                              'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
                              'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
    #duration = pd.DataFrame(df['YrSold'] - df['YearBuilt'], index=np.arange(df.shape[0]), columns=['Duration'])

    for num_col in numerical_features_idx:
        df[num_col] = df[num_col].fillna(-1)
    numerical_features = df[numerical_features_idx]
    #numerical_features = pd.concat([df[numerical_features_idx], duration], axis=1)

    # Handle Features that need further modified
    modified_features_idx = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'CentralAir', 'KitchenQual',
                             'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC']
    for modified_col in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu',
                         'GarageQual', 'GarageCond', 'PoolQC']:
        df[modified_col] = df[modified_col].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0})
    df['CentralAir'] = df['CentralAir'].map({'Y': 1, 'N': 0})
    df['GarageFinish'] = df['GarageFinish'].map({'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0})
    df['PavedDrive'] = df['PavedDrive'].map({'Y': 2, 'P': 1, 'N': 0})
    modified_features = df[modified_features_idx]
    # Handle Categorical Features
    categorical_features_idx = np.setdiff1d(df.columns, numerical_features_idx + modified_features_idx + ['Id', 'SalePrice',
                                                                                                          'MSSubClass'])
    categorical_features = pd.concat([pd.get_dummies(df['MSSubClass']), pd.get_dummies(df[categorical_features_idx])], axis=1)
    output_features = pd.concat([numerical_features, modified_features, categorical_features], axis=1)
    return output_features

if __name__ == '__main__':
    output_opt = argv[1]
    train_test = argv[2]
    if train_test == '-tr':
        df = pd.read_csv('data/train.csv')
        y = df['SalePrice']
        output_features = modify_data(df)
        output_features.columns = output_features.columns.astype(str)
        with open('train_idx', 'w+') as idx_file:
            line = ''
            for i in output_features.columns:
                line += '{0},'.format(i)
            line = line[:-1]
            idx_file.write(line)
    elif train_test == '-te':
        train_idx_file = argv[3]
        df = pd.read_csv('data/test.csv')
        output_features = modify_data(df)
        with open(train_idx_file, 'r') as idx_file:
            train_idx = idx_file.read()
        idx = train_idx.split(',')
        diff = np.setdiff1d(idx, list(map(str, output_features.columns.tolist())))
        num_sample = output_features.shape[0]
        empty = pd.DataFrame(0, index=np.arange(num_sample), columns=diff)
        output_features = pd.concat([output_features, empty], axis=1)
        output_features.columns = output_features.columns.astype(str)
        output_features = output_features[idx]
    else:
        raise IOError
    (sample_num, features_num) = output_features.shape

    if output_opt == '-svm':
        train_sample_num = int(0.8 * sample_num)
        test_sample_num = sample_num - train_sample_num
        with open('svm_train_data', mode='w+') as output_file:
            for sample in range(train_sample_num):
                current_line = '{0}'.format(y[sample])
                for feature in range(features_num):
                    current_line += '\t{0}:{1}'.format(feature+1, output_features.iloc[sample, feature])
                current_line += '\n'
                output_file.write(current_line)
        with open('svm_valid_data', mode='w+') as output_file:
            for sample in range(train_sample_num, sample_num):
                current_line = ''
                for feature in range(features_num):
                    current_line += '{0}:{1}\t'.format(feature+1, output_features.iloc[sample, feature])
                current_line += '\n'
                output_file.write(current_line)
    elif output_opt == '-csv':
        if train_test == '-tr':
            output_features = pd.concat([y, output_features], axis=1)
            output_features.to_csv('input_train_data.csv')
        elif train_test == '-te':
            output_features.to_csv('input_test_data.csv')
    elif output_opt == '-df':
        if train_test == '-tr':
            output_features = pd.concat([y, output_features], axis=1)
            output_features.to_pickle('input_train_data')
        elif train_test == '-te':
            output_features.to_pickle('input_test_data')

