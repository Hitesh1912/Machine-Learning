#this code runs linear regression on spamdata set

import numpy as np
import pandas as pd
from numpy.linalg import inv
import time
from random import randrange

# Function importing Dataset
def importdata():
    data = pd.read_csv(
        'http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data',
        sep=',', header=None)
    # print("Dataset Shape: ", data.shape)
    # #for missing values remove row or column with (atleast 1) any Nan Null present
    data = data.dropna(axis=0, how='any')
    return data.values


# Function to split the dataset
def splitdataset(data):
    # Seperating the target variable
    x = [row[0:len(row) - 1] for row in data]
    y = [row[-1] for row in data]
    return x, y


#remove test set from training set
def removearray(train,arr):
    ind = 0
    size = len(train)
    while ind != size and not np.array_equal(train[ind],arr):
        ind += 1
    if ind != size:
        train.pop(ind)
    else:
        raise ValueError('not found')


# Calculate accuracy
def accuracy_val(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def get_best_param(X, y):
    X_transpose = X.T
    # normal equation
    # theta_best = (X.T * X)^(-1) * X.T * y
    best_params = inv(X_transpose.dot(X)).dot(X_transpose).dot(y)
    return np.array(best_params)  # returns a list


# test prediction
def predict(X_test,params):
    # 920 #size of xtest
    X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]
    # y = h_Theta_X(Theta) = Theta.T * X
    prediction = X_test_b.dot(params)
    prediction = classify_prediction(prediction)
    return prediction


def classify_prediction(prediction):
    theta = np.mean(prediction)
    new_prediction = list()
    for score in prediction:
        if score < theta:
            new_prediction.append(0)
        else:
            new_prediction.append(1)
    return new_prediction


def linear_reg(train_set,test_set):
    X, y = splitdataset(train_set)
    X_test, y_test = splitdataset(test_set)
    X_b = np.c_[np.ones((len(X), 1)), X]  # set bias term to 1 for each sample
    params = get_best_param(X_b, y)
    prediction = predict(X_test, params)
    return prediction

def k_fold_split(dataset, k_folds):
    dataset_split = list()
    dataset_1 = list(dataset)
    fold_size = int(len(dataset) / k_folds)
    for i in range(k_folds):
        fold = list()
        while len(fold) < fold_size:
            idx = randrange(len(dataset_1))
            fold.append(dataset_1.pop(idx))
        dataset_split.append(fold)
    return dataset_split


def evaluate_model(dataset,k_folds):
    folds = k_fold_split(dataset, k_folds)
    scores = list()
    for fold in folds:
        test_set = list()
        train_set = list(folds)
        removearray(train_set, fold)
        train_set = sum(train_set, [])
        for r in fold:
            row = list(r)
            test_set.append(row)
            row[-1] = None
        predicted = linear_reg(train_set,test_set)
        actual = [r[-1] for r in fold]
        acc = accuracy_val(actual, predicted)
        scores.append(acc)
    return scores



if __name__ == '__main__':
    s = time.time()
    data_set = importdata()
    accuracy = evaluate_model(data_set, 5)
    print('Accuracy: %s' % accuracy)
    print('Mean Accuracy: %.3f%%' % (round(sum(accuracy) / float(len(accuracy)))))
    e = time.time()
    print("time",e-s)




