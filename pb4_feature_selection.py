

#<Author > HITESH VERMA

import numpy as np
import pandas as pd
import time, math
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
    # print(np.shape(x),np.shape(y))
    return x, y

def get_accuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return ( correct /float(len(testSet))) * 100.0

def euclidean_distance(instance1, instance2):
    dst = distance.euclidean(instance1, instance2)
    return dst

def vote(neighbor_labels):
    """ Return the most common class among the neighbor samples """
    counts = np.bincount(neighbor_labels.astype('int'))
    return counts.argmax()

def seperated_by_class(X,y):
    group = {}
    y = y.astype(int)
    y = y.reshape(len(y))
    classes = list(set(y))
    for c in classes:
        class1_indices = np.where(y == c)[0]
        group[c] = X[class1_indices]
    return group


def predict(X_test, X_train, y_train,y_test, k):
    y_pred = np.empty(X_test.shape[0])
    # Determine the class of each sample
    for i, test_sample in enumerate(X_test):
        # Sort the training samples by their distance to the test sample and get the K nearest
        idx = np.argsort([euclidean_distance(test_sample, x) for x in X_train])[:k]
        k_nearest_neighbors = np.array([y_train[i] for i in idx])
        # Label sample as the most common class label
        k_nearest_neighbors = np.ravel(k_nearest_neighbors)
        y_pred[i] = vote(k_nearest_neighbors)
        # print("prediction: ",y_pred[i],"actual:  ",y_test)
    return y_pred


def get_features(X, y):
    k = 5
    # group = seperated_by_class(X_train, y_train)
    w = np.zeros(np.shape(X)[1],)
    for i, z in enumerate(X):
        same_idx = np.where(y == y[i])[0]
        opp_idx = np.where(y != y[i])[0]
        closest_same_idx = np.argsort([euclidean_distance(z, X[j]) for j in same_idx])
        closest_same_idx1 = closest_same_idx[1]
        closest_opp_idx = np.argsort([euclidean_distance(z, X[j]) for j in opp_idx])
        closest_opp_idx1 = closest_opp_idx[0]
        w = w - ((X[closest_same_idx1] - X[i]) ** 2) + ((X[closest_opp_idx1] - X[i])  ** 2)
    features = np.argsort(w)[::-1]
    return features[:k]




def main():
    # print("random",i)
    start = time.time()
    # np.random.seed(1)
    # spam dataset =======================================================
    dataset = importdata('http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data')
    # np.random.shuffle(dataset)
    X,y = splitdataset(dataset)
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    # print("normalization done")

    #run to get selected features
    # features = get_features(X, y)
    # print("selected features",features)
    #selected features [ 3 17 16 54 21]


    X = np.column_stack((X[:,3],X[:,17],X[:,16],X[:,54],X[:,21]))
    # print("x",np.shape(X))
    X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=15)
    # print(np.shape(X_train), np.shape(y_train))
    # print(np.shape(X_test), np.shape(y_test))
    k = 7
    predictions = predict(X_test, X_train, y_train, y_test, k)
    accuracy = metrics.accuracy_score(y_test, predictions)
    print('Accuracy:', accuracy)


    # end = time.time()
    # print("time",end - start)


if __name__ == '__main__':
    main()
    # for i in range(1,20):
    #     main(i)  #15