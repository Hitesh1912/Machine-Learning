# Q6 Implementing Neural network using Max likelihood (cross entropy) loss using softmax

import numpy as np
import pandas as pd
from random import seed



# Function importing Dataset
def importdata():
    global train_data
    train_data = pd.read_csv(
        'train_wine.csv', sep =',', header=None)
    print(train_data.shape)
    test_data = pd.read_csv(
        'test_wine.csv', sep =',', header=None)
    print(test_data.shape)
    return train_data.values, test_data.values


# Function to split the dataset
def splitdataset(data):
    # Seperating the target variable
    x = data[:, 1:data.shape[1]]
    y = data[:, 0]
    print(np.shape(x), np.shape(y))
    return x, y

def feature_normalization(x):
    mu = np.mean(x,axis=0)
    sigma = np.std(x,axis=0)
    return mu, sigma

def normalization(x,mu,sigma):
    x = np.subtract(x, mu)
    x = np.divide(x, sigma)
    return x

def dense_to_one_hot(y):
    y_d = pd.get_dummies(y.flatten())  #one-hot
    y_d = y_d.values
    y_d = y_d.astype('float')
    return y_d

# Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))


def stable_softmax(x):
    exps = np.exp(x)
    exps = exps / np.sum(exps, axis=1, keepdims= True)
    return exps


#y_true is one-hot ecnoded
def cross_entropy(y_true, predicted_prob):
    m = len(predicted_prob)
    log_likelihood =  -1 * y_true * np.log(predicted_prob)
    return np.sum(log_likelihood) / m


#y_true is hot encoded
def delta_cross_entropy(y_true, p):
    prob = p.copy()
    m = y_true.shape[0]
    prob = p - y_true
    prob = prob/m
    return prob


#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)


def evaluate_model(X,Y,X_test,y_test,inputlayer_neurons,hiddenlayer_neurons,output_neurons,lr,epochs):
    # seed = 128
    # weight and bias initialization
    wh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
    bh = np.random.uniform(size=(1, hiddenlayer_neurons))
    wout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
    bout = np.random.uniform(size=(1, output_neurons))
    y_d = dense_to_one_hot(Y)

    for epoch in range(epochs):
        # Forward Propogation
        hidden_layer_input1 = np.dot(X, wh)
        hidden_layer_input = hidden_layer_input1 + bh
        hiddenlayer_activations = sigmoid(hidden_layer_input)
        output_layer_input1 = np.dot(hiddenlayer_activations, wout)
        output_layer_input = output_layer_input1 + bout
        output = stable_softmax(output_layer_input)

        # Backpropagation
        d_output = delta_cross_entropy(y_d,output)
        Error_at_hidden_layer = d_output.dot(wout.T)
        slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
        d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer

        wout -= hiddenlayer_activations.T.dot(d_output) * lr
        bout -= np.sum(d_output, axis=0, keepdims=True) * lr

        wh -= X.T.dot(d_hiddenlayer) * lr
        bh -= np.sum(d_hiddenlayer, axis=0, keepdims=True) * lr
        epoch_loss = cross_entropy(y_d, output)
        print('Epoch', epoch, 'loss:', epoch_loss)

    prediction(X_test,y_test,wh,bh,wout,bout)



def prediction(X_test,y_test,wh,bh,wout,bout):
    hidden_layer_input1 = np.dot(X_test, wh)
    hidden_layer_input = hidden_layer_input1 + bh
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input1 = np.dot(hiddenlayer_activations, wout)
    output_layer_input = output_layer_input1 + bout
    output = stable_softmax(output_layer_input)
    pred = np.round(output)

    ytest_d = dense_to_one_hot(y_test) #one-hot
    pred_temp = np.equal(np.argmax(pred, 1), np.argmax(ytest_d, 1))
    accuracy = np.mean(pred_temp.astype('float'))
    print("accuracy",accuracy * 100)



if __name__ == '__main__':
    #get dataset
    trainset, testset  = importdata()

    #split features, label
    X_train, y_train = splitdataset(trainset)
    X_test, y_test = splitdataset(testset)

    #feature normalization
    mu, sigma = feature_normalization(X_train)
    X_train = normalization(X_train, mu, sigma)
    X_test = normalization(X_test, mu, sigma)

    #hyperparameters
    epoch = 500
    lr = 0.1
    inputlayer_neurons = X_train.shape[1]  # number of features in data set
    hiddenlayer_neurons = 5  # number of hidden layers neurons
    output_neurons = 3  # number of neurons at output layer

    evaluate_model(X_train,y_train,X_test,y_test,inputlayer_neurons,hiddenlayer_neurons,output_neurons,lr,epoch)