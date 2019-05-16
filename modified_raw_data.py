import pandas as pd
import numpy as np
from sys import argv


output_opt = argv[1]

df = pd.read_csv('data/train.csv')
y = df['SalePrice']

# Handle Missing Data
df['MasVnrType'] = df['MasVnrType'].fillna('MING')
df['Electrical'] = df['Electrical'].fillna('SBrkr')

for num_col in ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']:
    df[num_col] = df[num_col].fillna(-1)

for cat_col in ['Alley', 'GarageType', 'Fence', 'MiscFeature', 'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2',
                'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'BsmtExposure']:
    df[cat_col] = df[cat_col].fillna('None')

# Numerical Features
numerical_features_idx = ['LotArea', 'LotFrontage', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
                      'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                      'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
                      'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
                      'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
numerical_features = df[numerical_features_idx]

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
(sample_num, features_num) = output_features.shape


if output_opt == '-svm':
    train_sample_num = int(0.6 * sample_num)
    test_sample_num = sample_num - train_sample_num
    with open('svm_train_data', mode='w+') as output_file:
        for sample in range(train_sample_num):
            current_line = '{0}'.format(y[sample])
            for feature in range(features_num):
                current_line += '\t{0}:{1}'.format(feature+1, output_features.iloc[sample, feature])
            current_line += '\n'
            output_file.write(current_line)
    with open('svm_test_data', mode='w+') as output_file:
        for sample in range(test_sample_num):
            current_line = '{0}'.format(y[sample])
            for feature in range(features_num):
                current_line += '\t{0}:{1}'.format(feature+1, output_features.iloc[sample, feature])
            current_line += '\n'
            output_file.write(current_line)
elif output_opt == '-csv':
    output_features.to_csv('train.csv')
elif output_opt == '-pd':
    train_sample_num = int(0.95 * sample_num)
    test_sample_num = sample_num - train_sample_num
    output_features.iloc[:train_sample_num, :].to_pickle('train_data')
    output_features.iloc[train_sample_num:, :].to_pickle('test_data')

