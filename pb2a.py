

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


def predict(X_test, X_train, y_train,y_test):
    y_pred = np.empty(X_test.shape[0])
    R = 2.5
    # Determine the class of each sample
    for i, test_sample in enumerate(X_test):
        # idx = [i if euclidean_distance(test_sample, x) < R else -1 for i, x in enumerate(X_train)]
        # idx = list(set(idx))
        # idx.remove(-1)
        # if not idx:
        #     idx = np.argsort([euclidean_distance(test_sample, x) for x in X_train])[:1]

        idx = [i if cosine_distance(test_sample, x) < 0.83 else -1 for i, x in enumerate(X_train)]
        idx = list(set(idx))
        idx.remove(-1)
        # if not idx:
        #     idx = np.argsort([euclidean_distance(test_sample, x) for x in X_train])[:1]
        # Extract the labels of the K nearest neighboring training samples
        k_nearest_neighbors = np.array([y_train[i] for i in idx])
        # Label sample as the most common class label
        k_nearest_neighbors = np.ravel(k_nearest_neighbors)
        y_pred[i] = vote(k_nearest_neighbors)
        # print("prediction: ",y_pred[i],"actual:  ",y_test)
    return y_pred



def sample_by_class(dataset):
    new_dataset = np.zeros((1, np.shape(dataset)[1]))
    y = dataset[:,-1]
    for i in range(10):
        idx = np.where(y.astype(int) == i)[0]
        temp_dataset = dataset[idx]
        sample = int(len(temp_dataset) * 0.5)
        new_sample = temp_dataset[:sample,:]
        new_dataset = np.concatenate((new_dataset,new_sample),axis=0)
    new_dataset = np.array(new_dataset)
    new_dataset = np.delete(new_dataset, 0, axis=0)
    print("Sampled dataset", new_dataset.shape)
    return new_dataset



def main():
    start = time.time()
    np.random.seed(1)
    #============================================================
    # print("running on spam dataset")
    # dataset = importdata('http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data')
    # # np.random.shuffle(dataset)
    # X,y = splitdataset(dataset)
    # scaler = StandardScaler()
    # scaler.fit(X)
    # X = scaler.transform(X)
    # print("normalization done")
    # X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.20) #.33
    # print(np.shape(X_train), np.shape(y_train))
    # print(np.shape(X_test), np.shape(y_test))

    # DIGIT DATASET ===============================================================
    print("running on digit dataset")
    train_dataset = pd.read_csv('digit/Haar_feature_full_training.csv', header=None, sep=',').values
    print(train_dataset.shape)
    test_dataset = pd.read_csv('digit/Haar_feature_testing.csv', header=None, sep=',').values
    print(test_dataset.shape)

    # # ==============sampling======================================
    train_dataset = sample_by_class(train_dataset)
    test_dataset = sample_by_class(test_dataset)
    # # ====================================================
    X_train, y_train = splitdataset(train_dataset)
    X_test, y_test = splitdataset(test_dataset)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    print("normalization done")
    print(np.shape(X_train), np.shape(y_train))
    print(np.shape(X_test), np.shape(y_test))
    # ===============================================

    predictions = predict(X_test, X_train, y_train,y_test)
    accuracy = metrics.accuracy_score(y_test,predictions)

    # accuracy = get_accuracy(testSet, predictions)
    print('Accuracy:',accuracy)
    end = time.time()
    print("time",end - start)


if __name__ == '__main__':
    main()