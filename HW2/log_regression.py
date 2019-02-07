#this code runs linear regression on spamdata set

import numpy as np
import pandas as pd
import time
from random import randrange, seed, shuffle
import math
import matplotlib.pyplot as plt
from sklearn import metrics

# Function importing Dataset
def importdata():
    data = pd.read_csv(
        '/MSCS/ML/code/HW2/spambase_csv.csv', sep=',', header=None)
    # data = pd.read_csv(
    #     'http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data',
    #     sep=',', header=None)
    # #for missing values remove row or column with (atleast 1) any Nan Null present
    # data = data.dropna(axis=0, how='any')
    print(data.shape)
    return data.values

# Function to split the dataset
def splitdataset(data):
    # Seperating the target variable
    x = data[:, 0:data.shape[1]-1]
    y = data[:, data.shape[1]-1:data.shape[1]]
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

def plot_roc(score, y):
    roc_x = []
    roc_y = []
    min_score = min(score)
    max_score = max(score)
    thr = np.linspace(min_score, max_score, 30)
    FP = 0
    TP = 0
    N = sum(y)
    P = len(y) - N
    for (i, T) in enumerate(thr):
        for i in range(0, len(score)):
            if (score[i] > T):
                if (y[i] == 1):
                    TP = TP + 1
                if (y[i] == 0):
                    FP = FP + 1
        roc_x.append(FP / float(N))
        roc_y.append(TP / float(P))
        FP = 0
        TP = 0
    # roc_x = FP
    # roc_y = TP
    plt.plot(roc_x, roc_y)
    plt.show()
    auc1 = metrics.auc(roc_x, roc_y)
    print("AUC score", auc1)



def confusion_matrix(actual, predicted):
    predicted = np.asarray(predicted).reshape(-1)
    y_actu = pd.Series(actual, name='Actual')
    y_pred = pd.Series(predicted, name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred)
    print(df_confusion)


# Calculate accuracy
def accuracy_val(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def sigmoid(z):
    z = np.array(z, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-z)))


# test prediction
def predict(X_test,params):
    X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]
    score = sigmoid(X_test_b.dot(params))
    prediction = np.round(score)
    return prediction, score


def feature_normalization(x):
    mu = np.mean(x,axis=0)
    sigma = np.std(x,axis=0)
    return mu, sigma


def normalization(x,mu,sigma):
    x = np.subtract(x, mu)
    x = np.divide(x, sigma)
    return x



def compute_loss(X, y, w):
    n = len(X)
    grad = np.zeros(np.shape(w))
    J = 0
    h_w = sigmoid(X.dot(w))
    J = sum(y.dot(np.log(h_w)) + (1 - y).dot(np.log(1 - h_w)))
    return J


def gradient_descent(X, y, w, alpha, num_iters):
    #initialize
    n = len(X)
    X_transpose = X.T
    h_w = sigmoid(X.dot(w))
    size = (num_iters, 1)
    J_history = np.zeros(size)
    for iter in range(num_iters):
        w = w + ((alpha / n ) * (X_transpose.dot(y - h_w)))
    return w, J_history


def log_reg(X, y, alpha, num_iters):
    X_b = np.c_[np.ones((len(X), 1)), X]  # set bias term to 1 for each sample
    size = (X_b.shape[1],1)
    w = np.random.normal(size=size)
    #model training
    params, J_history = gradient_descent(X_b, y, w, alpha, num_iters)
    return params


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


def evaluate_model(dataset, k_folds, alpha, num_iters):
    folds = k_fold_split(dataset, k_folds)
    test_scores = list()
    train_scores = list()
    for fold in folds:
        test_set = list()
        train_set = list(folds)
        removearray(train_set, fold)
        train_set = sum(train_set, [])
        for r in fold:
            row = list(r)
            test_set.append(row)
            row[-1] = None
        y_test = [r[-1] for r in fold]
        train_set = np.array(train_set)
        test_set = np.array(test_set)
        X, y = splitdataset(train_set)
        X_test, y_t = splitdataset(test_set)

        # #feature normalization:
        mu, sigma = feature_normalization(X)
        X = normalization(X, mu, sigma)
        X_test = normalization(X_test,mu, sigma)

        params = log_reg(X, y, alpha, num_iters)
        # after training model, make predictions
        test_predicted_label, score = predict(X_test, params)
        #plot roc
        plot_roc(score,y_test)
        #display confusion matrix
        confusion_matrix(y_test,test_predicted_label)

        acc = accuracy_val(y_test, test_predicted_label)
        test_scores.append(acc)

        train_predicted_label, tscore = predict(X, params)
        train_acc = accuracy_val(y, train_predicted_label)
        train_scores.append(train_acc)
    return test_scores, train_scores




if __name__ == '__main__':
    s = time.time()
    seed(1)
    alpha = 0.01
    num_iters = 1000
    dataset = importdata()
    shuffle(dataset)
    test_acc, train_acc = evaluate_model(dataset, 2, alpha, num_iters)
    print('Test Accuracy: %s' % test_acc)
    print('Test Acc: %.3f%%' % (sum(test_acc) / float(len(test_acc))))
    print('Train accuracy: %s' % train_acc)
    print('Train Acc: %.3f%%' % (sum(train_acc) / float(len(train_acc))))
    e = time.time()
    print("time",e-s)


