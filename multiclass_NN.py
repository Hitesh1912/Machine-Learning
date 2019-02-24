import tensorflow as tf
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

# Calculate accuracy
def accuracy_val(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def dense_to_one_hot(y):
    y_d = pd.get_dummies(y.flatten())  #one-hot
    y_d = y_d.values
    y_d = y_d.astype('float')
    return y_d

def compute_loss(y_true, y_pred):
    loss = np.mean(np.power(y_true - y_pred, 2))
    return loss


#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)


def evaluate_model(X,Y,X_test,y_test,inputlayer_neurons,hiddenlayer_neurons,output_neurons,lr,epochs):
    # seed = 128
    # weight and bias initialization
    wh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons)) #13 x
    bh = np.random.uniform(size=(1, hiddenlayer_neurons))
    wout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
    bout = np.random.uniform(size=(1, output_neurons))

    y_d = dense_to_one_hot(Y)# one-hot

    for epoch in range(epochs):
        # Forward Propogation
        hidden_layer_input1 = np.dot(X, wh)
        hidden_layer_input = hidden_layer_input1 + bh
        hiddenlayer_activations = sigmoid(hidden_layer_input)
        output_layer_input1 = np.dot(hiddenlayer_activations, wout)
        output_layer_input = output_layer_input1 + bout
        output = sigmoid(output_layer_input)

        # Backpropagation
        E = y_d - output
        slope_output_layer = derivatives_sigmoid(output)
        slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
        d_output = E * slope_output_layer
        Error_at_hidden_layer = d_output.dot(wout.T)
        d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer

        wout += hiddenlayer_activations.T.dot(d_output) * lr
        bout += np.sum(d_output, axis=0, keepdims=True) * lr

        wh += X.T.dot(d_hiddenlayer) * lr
        bh += np.sum(d_hiddenlayer, axis=0, keepdims=True) * lr

        pred = np.round(output)
        epoch_loss = compute_loss(y_d, pred)
        print('Epoch', epoch, 'loss:', epoch_loss)

    # print("training accuracy")
    # prediction(X,Y,wh,bh,wout,bout)
    print("testing accuracy")
    prediction(X_test,y_test,wh,bh,wout,bout)



def prediction(X_test,y_test,wh,bh,wout,bout):
    hidden_layer_input1 = np.dot(X_test, wh)
    hidden_layer_input = hidden_layer_input1 + bh
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input1 = np.dot(hiddenlayer_activations, wout)
    output_layer_input = output_layer_input1 + bout
    output = sigmoid(output_layer_input)
    pred = np.round(output)
    # print("pred",np.shape(pred))

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
    epoch = 35  # Setting training iterations
    lr = 0.1  # Setting learning rate
    inputlayer_neurons = X_train.shape[1]  # number of features in data set
    hiddenlayer_neurons = 5  # number of hidden layers neurons
    output_neurons = 3  # number of neurons at output layer

    evaluate_model(X_train,y_train,X_test,y_test,inputlayer_neurons,hiddenlayer_neurons,output_neurons,lr,epoch)