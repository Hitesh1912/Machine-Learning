import numpy as np
import pandas as pd


# Function importing Dataset
def importdata():
    data = pd.read_csv(
        '/MSCS/ML/CS6140_Code/HW3/dataset.csv', sep =',', header=None)
    print(data.shape)
    return data.values


def compute_loss(y_true, y_pred):
    loss = np.mean(np.power(y_true - y_pred, 2))
    return loss


#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)


def evaluate_model(X,Y,inputlayer_neurons,hiddenlayer_neurons,output_neurons,lr,epochs):
    # seed = 128
    # weight and bias initialization
    wh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
    bh = np.random.uniform(size=(1, hiddenlayer_neurons))
    wout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
    bout = np.random.uniform(size=(1, output_neurons))

    for epoch in range(epochs):
        # Forward Propogation
        hidden_layer_input1 = np.dot(X, wh)
        hidden_layer_input = hidden_layer_input1 + bh
        hiddenlayer_activations = sigmoid(hidden_layer_input)
        output_layer_input1 = np.dot(hiddenlayer_activations, wout)
        output_layer_input = output_layer_input1 + bout
        output = sigmoid(output_layer_input)

        # Backpropagation
        E = Y - output
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
        pred_temp = np.equal(np.argmax(pred, 1), np.argmax(Y, 1))
        accuracy = np.mean(pred_temp.astype('float'))
        epoch_loss = compute_loss(Y, pred)
        print('Epoch', epoch, 'loss:', epoch_loss, 'acc:', accuracy)

        if accuracy == 1.0:
            break

    print("final hidden values::")
    print(hiddenlayer_activations)
    print("output")
    print(np.round(output))
    print("accuracy",accuracy * 100)



if __name__ == '__main__':
    #get dataset
    X = importdata()
    X = np.array(X)
    Y = X.copy()
    #hyperparameters
    epoch = 5000  # Setting training iterations
    lr = 0.1  # Setting learning rate
    inputlayer_neurons = X.shape[1]  # number of features in data set
    hiddenlayer_neurons = 3  # number of hidden layers neurons
    output_neurons = X.shape[1]  # number of neurons at output layer

    evaluate_model(X,Y,inputlayer_neurons,hiddenlayer_neurons,output_neurons,lr,epoch)