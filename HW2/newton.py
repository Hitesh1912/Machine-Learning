import numpy as np
import pandas as pd
import time
from random import randrange, seed, shuffle
from numpy.linalg import pinv



# Function importing Dataset
def importdata():
    data = pd.read_csv(
        '/MSCS/ML/code/HW2/spambase_csv.csv', sep=',', header=None)
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
    prediction = np.round(sigmoid(X_test_b.dot(params)))
    return prediction


def feature_normalization(x):
    mu = np.mean(x,axis=0)
    sigma = np.std(x,axis=0)
    return mu, sigma

def normalization(x,mu,sigma):
    x = np.subtract(x, mu)
    x = np.divide(x, sigma)
    return x


def newton_fit(X, y, n_epoch):
    m = len(X)
    X = np.c_[np.ones((len(X), 1)), X]  # set bias term to 1 for each sample
    size = (X.shape[1],1)
    #random initializations
    w = np.random.normal(size=size)
    for i in range(n_epoch):
        h = sigmoid(X.dot(w))
        error = h - y
        Sk = h * (1 - h)
        Sk = np.reshape(Sk,(len(Sk),))
        S = np.diag(Sk)
        H = X.T.dot(S.dot(X))
        jj = X.T.dot(error)
        grad = pinv(H).dot(jj)
        w = w - (0.01)* np.array(grad)
    return w


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


def evaluate_model(dataset, k_folds, n_epoch):
    folds = k_fold_split(dataset, k_folds)
    test_scores, train_scores = list(), list()

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

        params = newton_fit(X, y, n_epoch)

        # after training model, make predictions
        test_predicted_label = predict(X_test, params)
        acc = accuracy_val(y_test, test_predicted_label)
        test_scores.append(acc)

        train_predicted_label = predict(X, params)
        train_acc = accuracy_val(y, train_predicted_label)
        train_scores.append(train_acc)
    return test_scores, train_scores



if __name__ == '__main__':
    s = time.time()
    seed(1)
    num_iters = 100
    dataset = importdata()
    shuffle(dataset)
    test_acc, train_acc = evaluate_model(dataset, 4, num_iters)
    print('Test Accuracy: %s' % test_acc)
    print('Test Acc: %.3f%%' % (sum(test_acc) / float(len(test_acc))))
    print('Train accuracy: %s' % train_acc)
    print('Train Acc: %.3f%%' % (sum(train_acc) / float(len(train_acc))))
    e = time.time()
    print("time",e-s)


