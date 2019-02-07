#this code runs linear regression on spamdata set

import numpy as np
import pandas as pd
import time



# Function importing Dataset
def importdata():
    data = pd.read_csv(
        'http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/perceptronData.txt', sep='\t', header=None)
    print(data.shape)
    return data.values

# Function to split the dataset
def splitdataset(data):
    # Seperating the target variable
    x = data[:, 0:data.shape[1]-1]
    y = data[:, data.shape[1]-1:data.shape[1]]
    return x, y



def perceptron(X, Y, l_r, n_epoch):
    size = (X.shape[1],1)
    #random initialization
    w = np.random.normal(size=size)
    for epoch in range(n_epoch):
        mistakes = list()
        predict = X.dot(w)
        for i, val in enumerate(predict):
            if val <= 0.0:
                mistakes.append(X[i])
        for x in mistakes:
            x = np.mat(x)
            w = w + l_r * (x.T)
        print("iterations %d mistakes %d" % (epoch, len(mistakes)))
        if len(mistakes) == 0:
            break
    return w



def evaluate_model(dataset,l_rate, num_iters):
    trainset = np.array(dataset)
    # preprocessing
    for i in range(trainset.shape[0]):
        if trainset[i][trainset.shape[1]-1] < 0:
            trainset[i] = trainset[i] * (-1)
    X, y = splitdataset(trainset)
    X = np.c_[np.ones((len(X), 1)), X]  # set bias term to 1 for each sample
    weights = perceptron(X, y, l_rate, num_iters)
    print("weights",weights)
    total = sum(X.dot(weights))
    w0 = 1 - total
    w0 = - (w0)
    print(w0)
    #normalized weight
    norm_weights = list()
    for w in weights:
        norm_weights.append(w / w0)
    print("norm_weights",norm_weights)



if __name__ == '__main__':
    s = time.time()
    num_iters = 100
    l_rate = 0.001
    dataset = importdata()
    evaluate_model(dataset,l_rate, num_iters)
    e = time.time()
    print("time",e-s)