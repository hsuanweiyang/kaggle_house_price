from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sys import argv
import numpy as np
from xgboost import XGBRegressor
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

input_train_file = argv[1]
input_test_file = argv[2]
reg_opt = argv[3]
numerical_features_idx = ['LotArea', 'LotFrontage', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
                      'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                      'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
                      'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
                      'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

def norm(data):
    data[numerical_features_idx] = (data[numerical_features_idx]-data[numerical_features_idx].mean())/\
                                   (data[numerical_features_idx].max()-data[numerical_features_idx].min())
    return data



input_train_data = pd.read_pickle(input_train_file)
input_test_data = pd.read_pickle(input_test_file)
permutation = np.random.permutation(input_train_data.shape[0])
#input_train_data = input_train_data.iloc[permutation, :]

train_x = input_train_data.iloc[:972, 1:]
train_y = input_train_data.iloc[:972, 0]
valid_x = input_train_data.iloc[972:, 1:]
valid_y = input_train_data.iloc[972:, 0]



if reg_opt == 'xg':
    reg = XGBRegressor(learning_rate=0.1, max_depth=4, n_estimators=360, min_child_weight=1)
elif reg_opt == 'svr':
    reg = SVR()
elif reg_opt == 'rf':
    reg = RandomForestRegressor()
reg.fit(train_x, train_y)

valid_result =reg.predict(valid_x)

rmse = np.sqrt(np.mean(np.square(np.log(valid_result)-np.log(valid_y))))
print(rmse)

#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#    print(input_test_data)
new_reg = XGBRegressor(learning_rate=0.1, max_depth=4, n_estimators=360, min_child_weight=1)
new_reg.fit(input_train_data.iloc[:, 1:], input_train_data.iloc[:, 0])
predict_result = new_reg.predict(input_test_data)

with open('submit_{0}.csv'.format(reg_opt), 'w+') as output_file:
    output_file.write('Id,SalePrice\n')
    for i in range(1461, 2920):
        output_file.write('{0},{1}\n'.format(i, predict_result[i-1461]))
