

#<Author > HITESH VERMA

import numpy as np
import pandas as pd
import time, math
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import operator
from scipy.spatial import distance


def importdata(file):
    spam_data = pd.read_csv(file,sep=',', header=None).values
    print(spam_data.shape)
    return spam_data

# Function to split the dataset
def splitdataset(data):
    # Seperating the target variable
    x = data[:, 0:data.shape[1]-1]
    y = data[:, data.shape[1]-1:data.shape[1]]
    print(np.shape(x),np.shape(y))
    return x, y

def get_accuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return ( correct /float(len(testSet))) * 100.0


def vote(neighbor_labels):
     """ Return the most common class among the neighbor samples """
     counts = np.bincount(neighbor_labels.astype('int'))
     return counts.argmax()


def euclidean_distance(instance1, instance2):
    dst = distance.euclidean(instance1, instance2)
    return dst


# def cosine_distance(x1, x2):
#     dot_product = np.dot(x1, x2)
#     norm_x1 = np.linalg.norm(x1)
#     norm_x2 = np.linalg.norm(x2)
#     return dot_product / (norm_x1 * norm_x2)

def cosine_distance(xi,x):
    return distance.cosine(xi,x)

def polynomial_kernel(x1, x2, power=2, coef=1):
    px_z = (np.inner(x1, x2) + coef) ** power
    return px_z

def gaussian_kernel(xi,x,sigma=1.0):
    similarity = np.exp(- (distance.euclidean(xi, x)** 2) / (2 * (sigma ** 2)))
    return similarity




def predict(X_test, X_train, y_train,y_test, k):
    y_pred = np.empty(X_test.shape[0])
    # Determine the class of each sample
    for i, test_sample in enumerate(X_test):
        # Sort the training samples by their distance to the test sample and get the K nearest
        idx = np.argsort([euclidean_distance(test_sample, x) for x in X_train])[:k]
        # idx = np.argsort([gaussian_kernel(1.0, test_sample, x) for x in X_train])[:k]  #40
        # idx = np.argsort([cosine_distance(test_sample,x) for x in X_train])[:k]
        # idx = np.argsort([polynomial_kernel(2, test_sample, x) for x in X_train])[:k] #55
        # Extract the labels of the K nearest neighboring training samples
        k_nearest_neighbors = np.array([y_train[i] for i in idx])
        # Label sample as the most common class label
        k_nearest_neighbors = np.ravel(k_nearest_neighbors)
        y_pred[i] = vote(k_nearest_neighbors)
        # print("prediction: ",y_pred[i],"actual:  ",y_test)
    return y_pred




def main():
    start = time.time()
    np.random.seed(1)
    dataset = importdata('http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data')
    np.random.shuffle(dataset)
    X,y = splitdataset(dataset)
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    print("normalization done")
    X = np.column_stack((X[:, 3], X[:, 17], X[:, 16], X[:, 54], X[:, 21]))
    print("x", np.shape(X))
    X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.2) #.33
    print(np.shape(X_train), np.shape(y_train))
    print(np.shape(X_test), np.shape(y_test))

    k = 1
    predictions = predict(X_test, X_train, y_train,y_test, k)
    accuracy = metrics.accuracy_score(y_test,predictions)
    print('k ',k,'Accuracy:',accuracy)

    k = 3
    predictions = predict(X_test, X_train, y_train, y_test, k)
    accuracy = metrics.accuracy_score(y_test, predictions)
    print('k',k,'Accuracy:', accuracy)

    k = 7
    predictions = predict(X_test, X_train, y_train, y_test, k)
    accuracy = metrics.accuracy_score(y_test, predictions)
    print('k',k,'Accuracy:', accuracy)



    end = time.time()
    print("time",end - start)


if __name__ == '__main__':
    main()