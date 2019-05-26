import tensorflow as tf
import numpy as np
import pandas as pd
from sys import argv
from xgboost import XGBRegressor
import math
from sklearn.ensemble import RandomForestRegressor


def read_input_data(input_file):
    input_data = pd.read_pickle(input_file)
    y = input_data['SalePrice']
    x = pd.DataFrame.transpose(input_data.iloc[:, 1:])
    return x, y


def train_valid_split(input_x, input_y, train_portion=0.9):
    sample_num = input_x.shape[1]
    train_num = int(sample_num * train_portion)
    permutation = list(np.random.permutation(sample_num))
    shuffled_x = input_x.iloc[:, permutation]
    shuffled_y = input_y[permutation]
    train_x = shuffled_x.iloc[:, :train_num]
    valid_x = shuffled_x.iloc[:, train_num:]
    train_y = shuffled_y[:train_num]
    valid_y = shuffled_y[train_num:]
    return train_x, train_y, valid_x, valid_y


def create_placeholders(n_f):
    X = tf.placeholder(tf.float32, [n_f, None])
    Y = tf.placeholder(tf.float32, [None])
    return X, Y


def initialize_parameters(n_f):
    w1 = tf.get_variable('w1', [40, n_f], initializer=tf.keras.initializers.he_normal())
    b1 = tf.get_variable('b1', [40, 1], initializer=tf.zeros_initializer())
    w2 = tf.get_variable('w2', [60, 40], initializer=tf.keras.initializers.he_normal())
    b2 = tf.get_variable('b2', [60, 1], initializer=tf.zeros_initializer())
    w3 = tf.get_variable('w3', [30, 60], initializer=tf.keras.initializers.he_normal())
    b3 = tf.get_variable('b3', [30, 1], initializer=tf.zeros_initializer())
    w4 = tf.get_variable('w4', [20, 30], initializer=tf.keras.initializers.he_normal())
    b4 = tf.get_variable('b4', [20, 1], initializer=tf.zeros_initializer())
    parameters = {'w1': w1, 'b1': b1,
                  'w2': w2, 'b2': b2,
                  'w3': w3, 'b3': b3,
                  'w4': w4, 'b4': b4}
    return parameters


def add_layer(W, b, X, activation=tf.nn.leaky_relu):
    Z = tf.matmul(W, X) + b
    A = activation(Z)
    return A


def forward_propogation(X, parameters, drop_rate):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    w3 = parameters['w3']
    b3 = parameters['b3']
    w4 = parameters['w4']
    b4 = parameters['b4']

    A1 = add_layer(w1, b1, X)
    A1 = tf.nn.dropout(A1, rate=drop_rate)
    A2 = add_layer(w2, b2, A1)
    A2 = tf.nn.dropout(A2, rate=drop_rate)
    A3 = add_layer(w3, b3, A2)
    A3 = tf.nn.dropout(A3, rate=drop_rate)
    #A4 = add_layer(w4, b4, A3)
    #A4 = tf.nn.dropout(A4, rate=drop_rate)
    Z_out = tf.matmul(w4, A3) + b4
    return Z_out


def compute_cost(Y, Z_out):
    #Y = tf.log(Y)
    #Z_out = tf.log(tf.clip_by_value(Z_out, 1e-8, math.inf)) # clip the value to prevent predicting negative num
    loss = tf.math.sqrt(tf.losses.mean_squared_error(Y, tf.squeeze(Z_out)))
    return loss


def log_cost(Y, Z_out):
    Y = tf.log(Y)
    Z_out = tf.log(Z_out)
    loss = tf.math.sqrt(tf.losses.mean_squared_error(Y, tf.squeeze(Z_out)))
    return loss


def random_minibatch(x, y, minibatch_size=64, seed=0):
    n_sample = x.shape[1]
    mini_batches = []
    np.random.seed(seed)

    permutation = np.random.permutation(n_sample)
    shuffled_x = x.iloc[:, permutation]
    shuffled_y = y.iloc[permutation]

    num_batches = n_sample//minibatch_size

    for i in range(num_batches):
        mini_x = shuffled_x.iloc[:, i * minibatch_size:(i+1) * minibatch_size]
        mini_y = shuffled_y.iloc[i * minibatch_size:(i+1) * minibatch_size]
        mini_batches.append((mini_x, mini_y))

    if n_sample&minibatch_size != 0:
        mini_x = shuffled_x.iloc[:, num_batches * minibatch_size:]
        mini_y = shuffled_y.iloc[num_batches * minibatch_size:]
        mini_batches.append((mini_x, mini_y))

    return mini_batches


def model(train_x, train_y, valid_x, valid_y, learing_rate=0.001, num_epoch=10000, minibatch_size=128, dropout_rate=0.,
          regularization = 0., print_cost=True, draw=True):
    n_feature = train_x.shape[0]
    costs = []
    valid_costs = []

    # Train
    X, Y = create_placeholders(n_feature)
    parameters = initialize_parameters(n_feature)
    drop_rate = tf.placeholder(tf.float32, name='drop_rate')

    Z_out = forward_propogation(X, parameters, drop_rate)
    log_loss = log_cost(Y, Z_out)
    loss = compute_cost(Y, Z_out)
    if regularization > 0.:
        reg_loss = 0
        for key in ['w1', 'w2', 'w3', 'w4']:
            reg_loss += tf.reduce_sum(tf.square(parameters[key]))
        loss += reg_loss
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(learning_rate=learing_rate).minimize(loss)
    saver = tf.train.Saver(max_to_keep=5)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epoch):
            epoch_loss = 0.
            minibatces = random_minibatch(train_x, train_y, minibatch_size)
            num_minibatches = len(minibatces)

            for mini_batch in minibatces:
                (mini_x, mini_y) = mini_batch
                _, mini_loss = sess.run([optimizer, loss], feed_dict={X: mini_x, Y: mini_y, drop_rate: dropout_rate})
                epoch_loss += mini_loss/num_minibatches
            if (epoch+1)%100 == 0 and print_cost:
                print('epoch-{0}: {1}'.format(epoch+1, epoch_loss))
                print('Train:', log_loss.eval({X: train_x, Y: train_y, drop_rate: 0.}))
                print('Valid:', log_loss.eval({X: valid_x, Y: valid_y, drop_rate: 0.}))
                saver.save(sess, 'model_fc_e-{0}_lr-{1}_d-{2}/model_fc'.format(num_epoch, learing_rate, dropout_rate),
                           global_step=epoch+1)
        print('Train RMSE:', log_loss.eval({X: train_x, Y: train_y, drop_rate: 0.}))
        print('Valid RMSE:', log_loss.eval({X: valid_x, Y: valid_y, drop_rate: 0.}))
        predict = Z_out.eval({X:valid_x, Y:valid_y, drop_rate:0.})
        for i in range(10):
            print(valid_y.iloc[i], '\t', predict[0, i])
        parameters = sess.run(parameters)
    return parameters



if __name__ == '__main__':
    opt = argv[1]
    input_file = argv[2]
    if opt == '-tr':
        train_opt = argv[3:]
        epoch_num = 10000
        batch_size = 64
        learning_rate = 0.001
        regularization = 0.06



    features, label = read_input_data(input_file)
    X_train, Y_train, X_valid, Y_valid = train_valid_split(features, label)
    model(X_train, Y_train, X_valid, Y_valid, num_epoch=epoch_num, dropout_rate=0., regularization=regularization)