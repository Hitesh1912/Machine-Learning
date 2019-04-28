
#AUTHOR: HITESH VERMA

import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from sklearn.metrics import accuracy_score


def importdata(file):
    data = pd.read_csv(file, sep='\s+', header=None).values
    print(data.shape)
    return data

# Function to split the dataset
def splitdataset(data):
    # Seperating the target variable
    x = data[:, 0:data.shape[1]-1]
    y = data[:, data.shape[1]-1:data.shape[1]]
    return x, y

def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def gaussian_kernel(xi,x,sigma=1.0):
    similarity = np.exp(- (distance.euclidean(xi, x)** 2) / (2 * (sigma ** 2)))
    return similarity


def perceptron(X,y):
    m = np.zeros((X.shape[0],1))
    iter = 0
    while True:
        iter += 1
        mistakes = list()
        predict = np.sum(m * linear_kernel(X, X.T), axis=1)
        predict = np.sign(predict)
        for j, val in enumerate(predict):
            if val * y[j] <= 0.0:
                m[j] = m[j] + y[j]
                mistakes.append(j)
        print("iterations", iter, "mistakes", len(mistakes))
        if len(mistakes) == 0:
            break
    return m



def cal_kernel(X):
    kernel_val =[]
    for i in range(len(X)):
        kernel_val.append([gaussian_kernel(X[i], X[j]) for j in range(len(X))])
    return np.array(kernel_val)


def kernel_perceptron(X, y):
    m = np.zeros((X.shape[0], 1))
    iter = 0
    while True:
        mistakes = list()
        iter += 1
        kernel_val = cal_kernel(X) #1000 x 1
        predict = np.sum(m * kernel_val, axis=1)
        for j, val in enumerate(predict):
            if val * y[j] <= 0.0:
                m[j] = m[j] + y[j]
                mistakes.append(j)
        print("iterations", iter, "mistakes", len(mistakes))
        if len(mistakes) == 0:
            break
        return m



# def predict(X_train, X_test, y_train, y_test ,m):
#     prediction = np.ones(np.shape(X_test)[0],)
#     predict = np.sum(m * linear_kernel(X_test, X_test.T), axis=1)
#     for j, val in enumerate(predict):
#         if val * y_train[j] <= 0.0:
#             prediction[j] = -1
#     return prediction


def main():
    data = importdata('http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/perceptronData.txt')
    # data = importdata('http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/TwoSpirals/twoSpirals.txt')
    X, y = splitdataset(data)
    # X = np.c_[np.ones((len(X), 1)), X]  # set bias term to 1 for each sample
    m = perceptron(X, y)

    # m = kernel_perceptron(X,y)
    # print("m",m)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # .33
    # print(np.shape(X_train), np.shape(y_train))
    # print(np.shape(X_test), np.shape(y_test))
    #
    # prediction = predict(X_train, X_test, y_train, y_test ,m)
    # accuracy = accuracy_score(y_test, prediction)
    # print("accuracy",accuracy)



if __name__ == '__main__':
    main()
