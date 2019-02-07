# this code runs linear regression on housing price dataset

import numpy as np
import pandas as pd
import time
from random import shuffle
import matplotlib.pyplot as plt


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
    X = data[:, 0:data.shape[1]-1]
    Y = data[:, data.shape[1]-1:data.shape[1]]
    print(np.shape(X),np.shape(Y))
    return X, Y


def mean_squared_error(actual, predicted):
    mse = (np.square(np.array(actual) - np.array(predicted))).mean(axis=0)
    return mse


def feature_normalization(x):
    mu = np.mean(x, axis = 0)
    sigma = np.std(x, axis = 0)
    return mu, sigma


def normalization(x,mu,sigma):
    x = np.subtract(x, mu)
    x = np.divide(x, sigma)
    return x


# test prediction
def predict(X_test,params):
    X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]
    prediction = X_test_b.dot(params)
    return prediction



def gradient_descent(X, y, w, alpha, num_iters):
    # initialize
    n = len(X)
    X_transpose = X.T
    size = (num_iters, 1)
    J_history = np.zeros(size)
    for iter in range(num_iters):
        w = w - ((alpha / n) * (X_transpose.dot(X.dot(w) - y)))
        J_history[iter] = compute_error(X, y, w)
        print('iteration:%d J:%.3f' % (iter, J_history[iter]))
        if np.round((abs(J_history[iter - 1] - J_history[iter])), 3) <= 0.001 and iter > 0:
            break
    return w, J_history



def compute_error(X,y,w):
    n  = len(X)
    J = 0
    J = (1/ 2 * n) * sum(np.square(np.array(X.dot(w)) - np.array(y)))
    return J

def linear_reg(X,y, alpha, num_iters):
    X_b = np.c_[np.ones((len(X), 1)), X]  # set bias term to 1 for each sample
    size = (X_b.shape[1],1)
    #initialize parameter
    w = np.random.normal(size=size)
    params, J_history = gradient_descent(X_b, y, w, alpha, num_iters) #14 X 1
    # plot_graph(J_history, num_iters)
    return params


def evaluate_model(train_set, test_set, alpha, num_iters):
    X, y = splitdataset(train_set)
    X_test, y_test = splitdataset(test_set)
    score = 0.0
    # initialize mu and sigma
    # feature normalization
    mu, sigma = feature_normalization(X)
    #use updated mu and sigma value to normalize test set features
    X = normalization(X, mu, sigma)
    X_test = normalization(X_test, mu, sigma)

    params = linear_reg(X, y, alpha, num_iters)
    # after training model, make predictions
    train_predicted_value = predict(X, params)
    train_score = mean_squared_error(y, train_predicted_value)
    predicted_values = predict(X_test, params)
    score = mean_squared_error(y_test, predicted_values)
    return score, train_score



def plot_graph(J,num_iter):
    num = np.array(range(0, num_iter))
    J = np.concatenate(J)
    print(np.shape(J),np.shape(num))
    plt.plot(num, J, 'ro')
    plt.axis([0, num_iter, 0, max(J)])
    plt.ylabel('J(w)')
    plt.xlabel('# of iterations')
    plt.show()



if __name__ == '__main__':
    s = time.time()
    num_iters = 2000
    alpha = 0.01
    train_set, test_set = importdata()
    shuffle(train_set)

    test_score, train_score = evaluate_model(train_set, test_set, alpha,num_iters)
    print("test mse",test_score)
    print("train mse",train_score)
    e = time.time()
    print("time",e-s)




