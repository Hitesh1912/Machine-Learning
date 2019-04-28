
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
    print(np.shape(x),np.shape(y))
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

def cosine_distance(x1, x2):
    dot_product = np.dot(x1, x2)
    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)
    return dot_product / (norm_x1 * norm_x2)


def polynomial_kernel(x1, x2, power=2, coef=1):
    px_z = (np.inner(x1, x2) + coef) ** power
    return px_z

def gaussian_kernel(xi,x,sigma=1.0):
    similarity = np.exp(- (distance.euclidean(xi, x)** 2) / (2 * (sigma ** 2)))
    return similarity


def seperated_by_class(X,y):
    group = {}
    y = y.astype(int)
    y = y.reshape(len(y))
    classes = list(set(y))
    for c in classes:
        class1_indices = np.where(y == c)[0]
        group[c] = X[class1_indices]
    return group


def prior(X,y):
    m = np.shape(y)[0]
    p_y = {}
    group = seperated_by_class(X, y)
    for class_val, x_class in group.items():
        count_spam = len(x_class)
        p_y[class_val] = count_spam / m
    return p_y


def kernel_dense_est(X_train,y_train,test_sample,class_value):
    group = seperated_by_class(X_train, y_train)
    p_z = None
    m_c = len(group[class_value])
    # k_sum = np.sum([gaussian_kernel(test_sample, x) for x in group[class_value]])
    k_sum = np.sum([polynomial_kernel(test_sample, x) for x in group[class_value]])
    p_z = k_sum / m_c
    return p_z


#calculate class prob (0 or 1) for each data point
def cal_class_prob(X_train,y_train,test_sample, p_c):
    prob = {}
    for class_val, prior_c in p_c.items():
        prob[class_val] = kernel_dense_est(X_train,y_train,test_sample,class_val) * prior_c
    return prob



def predict(X_train,y_train,test_sample, p_c):
    class_probs = cal_class_prob(X_train,y_train,test_sample,p_c)
    bestLabel, bestProb = None, -1
    for class_value, probability in class_probs.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = class_value
    return bestLabel


def getPredictions(X_train,y_train,X_test, p_c):
    predictions = []
    for test_sample in X_test:
        result = predict(X_train,y_train, test_sample, p_c)
        predictions.append(result)
    return predictions


def sample_by_class(dataset):
    new_dataset = np.zeros((1, np.shape(dataset)[1]))
    y = dataset[:,-1]
    for i in range(10):
        idx = np.where(y.astype(int) == i)[0]
        temp_dataset = dataset[idx]
        sample = int(len(temp_dataset) * 0.2)
        new_sample = temp_dataset[:sample,:]
        new_dataset = np.concatenate((new_dataset,new_sample),axis=0)
    new_dataset = np.array(new_dataset)
    new_dataset = np.delete(new_dataset, 0, axis=0)
    print("Sampled dataset", new_dataset.shape)
    return new_dataset



def main():
    start = time.time()
    np.random.seed(1)
    # spam dataset =======================================================
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

    #DIGIT DATASET ===============================================================
    print("running on digit dataset")
    train_dataset = pd.read_csv('digit/Haar_feature_full_training.csv', header=None, sep=',').values
    print(train_dataset.shape)
    test_dataset = pd.read_csv('digit/Haar_feature_testing.csv', header=None, sep=',').values
    print(test_dataset.shape)
    # np.random.shuffle(train_dataset)
    # np.random.shuffle(test_dataset)
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
    #===============================================

    p_c = prior(X_train,y_train)
    predictions = getPredictions(X_train,y_train,X_test, p_c)
    accuracy = metrics.accuracy_score(y_test,predictions)

    # accuracy = get_accuracy(testSet, predictions)
    print('Accuracy:',accuracy)
    end = time.time()
    print("time",end - start)


if __name__ == '__main__':
    main()