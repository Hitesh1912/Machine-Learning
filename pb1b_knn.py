

#<Author > HITESH VERMA
import numpy as np
import pandas as pd
import time, math
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pb5.Haar_Features import load_mnist
from scipy.spatial import distance


def importdata(file):
    spam_data = pd.read_csv(file,sep=',', header=None).values
    #for missing values remove row or column with (atleast 1) any Nan Null present
    # spam_data = spam_data.dropna(axis=0, how='any')
    print(spam_data.shape)
    return spam_data

# Function to split the dataset
def splitdataset(data):
    # Seperating the target variable
    x = data[:, 0:data.shape[1]-1]
    y = data[:, -1]
    print(np.shape(x),np.shape(y))
    return x, y

def get_accuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return ( correct /float(len(testSet))) * 100.0



def vote(neighbor_labels):
     #Return the most common class among the neighbor samples
     counts = np.bincount(neighbor_labels.astype('int'))
     return counts.argmax()


# def cosine_distance(x1, x2):
#     dot_product = np.inner(x1, x2) #np.dot
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


def linear_kernel(x1, x2):
    return np.inner(x1, x2)


def predict(X_test, X_train, y_train,y_test, k):
    y_pred = np.empty(X_test.shape[0])
    # Determine the class of each sample
    print("running gaussian rbf kernel")
    # print("running cosine distance")
    # print("running poly kernel")
    for i, test_sample in enumerate(X_test):
        # Sort the training samples by their distance to the test sample and get the K nearest
        # idx = np.argsort([- polynomial_kernel(test_sample, x,2) for x in X_train])[:k]
        idx = np.argsort([- gaussian_kernel(test_sample, x) for x in X_train])[:k]
        # idx = np.argsort([cosine_distance(test_sample,x) for x in X_train])[:k]
        k_nearest_neighbors = np.array([y_train[i] for i in idx])
        # Label sample as the most common class label
        k_nearest_neighbors = np.ravel(k_nearest_neighbors)
        y_pred[i] = vote(k_nearest_neighbors)
        # print("prediction: ",y_pred[i],"actual:  ",y_test[i])
    return y_pred


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
    print("new_dataset", new_dataset.shape)
    return new_dataset



def main():
    start = time.time()
    np.random.seed(1)

    train_dataset = pd.read_csv('digit/Haar_feature_full_training.csv', header=None, sep=',').values
    print(train_dataset.shape)
    test_dataset = pd.read_csv('digit/Haar_feature_testing.csv', header=None, sep=',').values
    print(test_dataset.shape)
    # # ==============sampling======================================
    train_dataset = sample_by_class(train_dataset)
    test_dataset = sample_by_class(test_dataset)
    # # ====================================================
    # np.random.shuffle(train_dataset)
    # np.random.shuffle(test_dataset)
    X_train, y_train = splitdataset(train_dataset)
    X_test, y_test = splitdataset(test_dataset)

    # X_train = np.array(X_train)
    # X_test = np.array(X_test)
    # y_train = np.array(y_train)
    # y_test = np.array(y_test)
    # y_train = y_train.astype("int")
    # y_test = y_test.astype("int")

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    print("normalization done")

    k = 1
    predictions = predict(X_test, X_train, y_train,y_test, k)
    accuracy = metrics.accuracy_score(y_test,predictions)
    # accuracy = get_accuracy(testSet, predictions)
    print('k',k,'Accuracy:',accuracy)

    k = 3
    predictions = predict(X_test, X_train, y_train,y_test, k)
    accuracy = metrics.accuracy_score(y_test,predictions)
    # accuracy = get_accuracy(testSet, predictions)
    print('k', k, 'Accuracy:', accuracy)

    k = 7
    predictions = predict(X_test, X_train, y_train,y_test, k)
    accuracy = metrics.accuracy_score(y_test,predictions)
    # accuracy = get_accuracy(testSet, predictions)
    print('k', k, 'Accuracy:', accuracy)




    end = time.time()
    print("time",end - start)


if __name__ == '__main__':
    main()