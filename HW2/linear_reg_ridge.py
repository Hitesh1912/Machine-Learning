# this code runs linear regression on housing price dataset

import numpy as np
import pandas as pd
from numpy.linalg import inv
import time


# Function importing Dataset
def importdata():
    train = pd.read_csv(
        'http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/housing_train.txt',
        sep='\s+', header=None)
    test = pd.read_csv(
        'http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/housing_test.txt',
        sep='\s+', header=None)
    return train.values, test.values


# Function to split the dataset
def splitdataset(data):
    # Seperating the target variable
    X = data[:, 0:data.shape[1]-1]
    Y = data[:, data.shape[1]-1]
    return X, Y


def mean_squared_error(actual, predicted):
    mse = (np.square(np.array(actual) - np.array(predicted))).mean(axis=None)
    return mse

def get_best_param(X, y, learning_rate):
    X_transpose = X.T
    best_params = inv(X_transpose.dot(X) + learning_rate * (np.identity(X.shape[1]))).dot(X_transpose).dot(y)
    return np.array(best_params)

# test prediction
def predict(X_test,params):
    X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]
    prediction = X_test_b.dot(params)
    return prediction

def linear_reg(X, y, alpha):
    X_b = np.c_[np.ones((len(X), 1)), X]  # set bias term to 1 for each sample
    params = get_best_param(X_b, y, alpha)
    return params


def evaluate_model(train_set, test_set, lr):
    X, y = splitdataset(train_set)
    X_test, y_test = splitdataset(test_set)
    test_score = 0.0
    train_score = 0.0
    params = linear_reg(X, y,lr)
    #after training
    #predict for train
    train_predicted_values = predict(X, params)
    train_score = mean_squared_error(y, train_predicted_values)
    #predict for test
    predicted_values = predict(X_test, params)
    test_score = mean_squared_error(y_test, predicted_values)
    return test_score, train_score


if __name__ == '__main__':
    s = time.time()
    train_set, test_set = importdata()
    alpha = 0.01
    # regularized_rate = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10]
    test_score, train_score = evaluate_model(train_set, test_set, alpha)
    print("train score", train_score)
    print("test score",test_score)
    e = time.time()
    print("time",e-s)




