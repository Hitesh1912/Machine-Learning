import tensorflow as tf
import numpy as np
import pandas as pd
from random import seed


# Function importing Dataset
def importdata():
    data = pd.read_csv(
        '/MSCS/ML/CS6140_Code/HW3/dataset.csv', sep =',', header=None)
    print(data.shape)
    return data.values


def evaluate_model(X_train, epochs):
    tf.set_random_seed(1234)
    with tf.Session() as sess:
        # Initialize the variables (i.e. assign their default value)
        sess.run(tf.global_variables_initializer())
        # Training
        for epoch in range(epochs):
            # prev = epoch_loss
            _, cost, accuracy_val = sess.run([optimizer,loss, accuracy], feed_dict={input_layer: X_train, real_output: X_train})
            epoch_loss = cost
            print('Epoch', epoch, '/', epochs, 'loss:', epoch_loss, 'acc:',accuracy_val)
            if accuracy_val == 1:
                break

        encoded_value = np.round(sess.run(hidden_layer, feed_dict={input_layer: X_train}), 3)
        # print(encoded_value)


if __name__ == '__main__':
    #get dataset
    x_train = importdata()
    # Training Parameters
    learningrate = 0.1
    epoch =  300

    # Network Parameters
    num_input = x_train.shape[1]  #features
    num_hidden = 3

    # set weight and bias
    hidden_layer_weights = tf.Variable(tf.random_normal([num_input, num_hidden])) #8x3
    # hidden_layer_biases = tf.Variable(tf.random_normal([num_hidden]))
    hidden_layer_biases = tf.Variable(tf.zeros([num_hidden]))
    output_layer_weights = tf.Variable(tf.random_normal([num_hidden, num_input])) #3x8
    # output_layer_biases = tf.Variable(tf.random_normal([num_input]))
    output_layer_biases = tf.Variable(tf.zeros([num_input]))


    # neural network
    input_layer = tf.placeholder('float', [None, num_input])
    hidden_layer = tf.nn.sigmoid(tf.add(tf.matmul(input_layer, hidden_layer_weights), hidden_layer_biases))
    output_layer = tf.matmul(hidden_layer, output_layer_weights) + output_layer_biases
    real_output = tf.placeholder('float', [None, num_input])

    loss = tf.reduce_mean(tf.square(real_output - output_layer))

    pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(real_output, 1))
    accuracy = tf.reduce_mean(tf.cast(pred_temp, 'float'))

    optimizer = tf.train.AdamOptimizer(learning_rate = learningrate).minimize(loss)

    # #training
    evaluate_model(x_train, epoch)