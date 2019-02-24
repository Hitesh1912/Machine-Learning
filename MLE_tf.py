# Q6 Implementing Neural network using Max likelihood (cross entropy) loss using softmax

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
    # print(np.shape(x), np.shape(y))
    return x, y

def feature_normalization(x):
    mu = np.mean(x,axis=0)
    sigma = np.std(x,axis=0)
    return mu, sigma

def normalization(x,mu,sigma):
    x = np.subtract(x, mu)
    x = np.divide(x, sigma)
    return x

def dense_to_one_hot(labels_dense, num_classes=3):
    labels_dense = np.subtract(labels_dense,1)
    labels_one_hot = tf.one_hot(labels_dense,depth=3)
    return labels_one_hot.eval()



def evaluate_model(X_train, X_test, y_train, y_test, epochs, batch_size):
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # Initialize the variables (i.e. assign their default value)
        sess.run(init)
        for epoch in range(epochs):
            avg_cost = 0.0
            total_batch = int(len(X_train) / batch_size)
            x_batches = np.array_split(X_train, total_batch)
            y_batches = np.array_split(y_train, total_batch)
            for i in range(total_batch):
                batch_x, batch_y = x_batches[i], y_batches[i]
                batch_y = dense_to_one_hot(y_batches[i])
                _, c = sess.run([optimizer, loss], feed_dict={ input_layer: batch_x, real_output: batch_y})
                avg_cost += c / total_batch
            if epoch % 100 == 0:
                print("Epoch:", '%04d' % (epoch + 1), "loss=","{:.9f}".format(avg_cost))

        print("\nTraining complete!")

        # #prediction on test set
        predict = tf.argmax(output_layer, 1)
        pred = predict.eval({input_layer: X_test.reshape(-1, num_input)})
        print(pred)
        correct_prediction = np.add(pred,1)
        print(correct_prediction)

        pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(real_output, 1))
        accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
        print("Test Accuracy:", accuracy.eval({input_layer: X_test.reshape(-1, num_input), real_output: dense_to_one_hot(y_test)}))



if __name__ == '__main__':
    # To stop potential randomness
    seed = 128
    rng = np.random.RandomState(seed)

    #get dataset
    trainset, testset  = importdata()

    #split features, label
    X_train, y_train = splitdataset(trainset)
    X_test, y_test = splitdataset(testset)

    #feature normalization
    mu, sigma = feature_normalization(X_train)
    X_train = normalization(X_train, mu, sigma)
    X_test = normalization(X_test, mu, sigma)

    # Network Parameters
    num_input = X_train.shape[1]  #features 12
    num_hidden = 5
    num_output = 3

    # define placeholders
    input_layer = tf.placeholder(tf.float32, [None, num_input])
    real_output = tf.placeholder(tf.float32, [None, num_output])

    # Training Parameters
    learning_rate = 0.01
    epochs =  1000
    batch_size = 50

    # define weights and biases of the neural network
    hidden_layer_weights = tf.Variable(tf.random_normal([num_input, num_hidden], seed = seed))
    hidden_layer_biases = tf.Variable(tf.random_normal([num_hidden],seed = seed))
    output_layer_weights = tf.Variable(tf.random_normal([num_hidden, num_output],seed = seed))
    output_layer_biases = tf.Variable(tf.random_normal([num_output],seed = seed))


    # create our neural networks computational graph
    hidden_layer = tf.add(tf.matmul(input_layer, hidden_layer_weights), hidden_layer_biases)
    hidden_layer = tf.nn.relu(hidden_layer)
    output_layer = tf.matmul(hidden_layer, output_layer_weights) + output_layer_biases


    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer,labels=real_output)) # used in maximum likelihood
    #our backpropogation algorithm | ADAM is variant of Gradient Descent algorithm
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

    # #training
    evaluate_model(X_train, X_test, y_train, y_test, epochs, batch_size)