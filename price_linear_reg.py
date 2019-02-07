# this code runs linear regression on housing price dataset

import numpy as np
import pandas as pd
from numpy.linalg import inv
import time
import operator


# Function importing Dataset
def importdata():
    train = pd.read_csv(
        'http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/housing_train.txt',
        sep='\s+', header=None)
    test = pd.read_csv(
        'http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/housing_test.txt',
        sep='\s+', header=None)
    print("Dataset Shape: ", test.shape)
    return train.values, test.values


# Function to split the dataset
def splitdataset(data):
    # Seperating the target variable
    X = data[:, 0:data.shape[1]-2]
    Y = data[:, data.shape[1]-1]
    return X, Y


def mean_squared_error(actual, predicted):
    mse = (np.square(np.array(actual) - np.array(predicted))).mean(axis=None)
    return mse


def get_best_param(X, y, learning_rate):
    # learning_rate = 0.0444
    X_transpose = X.T
    # normal equation
    # theta_best = (X.T * X)^(-1) * X.T * y
    # best_params = inv(X_transpose.dot(X)).dot(X_transpose).dot(y)
    best_params = inv(X_transpose.dot(X) + learning_rate * (np.identity(13))).dot(X_transpose).dot(y)
    # print(np.array(best_params))
    return np.array(best_params)  # returns a list


# test prediction
def predict(X_test,params):
    X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]
    prediction = X_test_b.dot(params)
    # y = h_Theta_X(Theta) = Theta.T * X)
    return prediction



def linear_reg(X,y, X_test, learning_rate):
    X_b = np.c_[np.ones((len(X), 1)), X]  # set bias term to 1 for each sample
    params = get_best_param(X_b, y, learning_rate)
    prediction = predict(X_test, params)
    return prediction


def evaluate_model(train_set, test_set, lr):
    X, y = splitdataset(train_set)
    X_test, y_test = splitdataset(test_set)
    score = 0.0
    # scores = {}
    # for lr in learning_rate:
    #     predicted_values = linear_reg(X, y, X_test, lr)
    #     score = mean_squared_error(y_test, predicted_values)
    #     scores[lr] = score
    predicted_values = linear_reg(X, y, X_test, lr)
    score = mean_squared_error(y_test, predicted_values)
    return score


if __name__ == '__main__':
    s = time.time()
    train_set, test_set = importdata()
    # X, y = splitdataset(train_set)
    learning_rate = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10]
    # chose learning_rate as 1e-15 based on max score on cross-validation set
    score = evaluate_model(train_set, test_set, learning_rate[0])
    # score = evaluate_model(train_set, train_set[:150,:],learning_rate)
    # score = max(score.items(), key=operator.itemgetter(1))[0]

    print("score",score)
    e = time.time()
    print("time",e-s)




