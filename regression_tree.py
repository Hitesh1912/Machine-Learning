import numpy as np
import pandas as pd
import time

# importing Dataset
def importdata():
    train = pd.read_csv(
        'http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/housing_train.txt',
        sep='\s+', header=None)
    test = pd.read_csv(
        'http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/housing_test.txt',
        sep='\s+', header=None)
    return train.values, test.values


def error_gain(groups,label):
    left = groups[0]
    right = groups[1]
    left_label = list(set(row[-1] for row in left))
    right_label = list(set(row[-1] for row in right))
    left = (np.var(left_label) * (len(left_label)/len(label)) if len(left_label) != 0 else 0 )
    right = (np.var(right_label) * (len(right_label)/len(label)) if len(right_label) != 0 else 0)
    total = np.var(label)
    new_error_1 = left + right
    new_error = total - new_error_1
    return new_error


def mean_squared_error(actual, predicted):
    mse = (np.square(np.array(actual) - np.array(predicted))).mean(axis=None)
    return mse


def make_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] <= value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Select the best split point for a dataset
def best_split(dataset):
    #Get Y label list
    y_values = list(set(row[-1] for row in dataset))
    split_feature, threshold, split_error, split_groups = 0, 0.0, 0.0, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = make_split(index, row[index], dataset)
            error = error_gain(groups, y_values)
            if error > split_error:
                split_feature, threshold, split_error, split_groups = index, row[index], error, groups
    print('X%d: %.3f error_gain=%.3f' % ((split_feature + 1), threshold, split_error))
    return {'index':split_feature, 'value':threshold, 'groups':split_groups, 'gain':split_error}

def predict(node, test_row):
    if test_row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], test_row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], test_row)
        else:
            return node['right']


def terminal_node(node):
    outcomes = [row[-1] for row in node]
    # mean of the y labels in the group(left or right)
    return np.mean(outcomes)


# Create child splits for the node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    # check if left or right group of rows is empty
    if not left or not right:
        node['left'] = node['right'] = terminal_node(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = terminal_node(left), terminal_node(right)
        return
    if len(left) <= min_size:
        node['left'] = terminal_node(left)
    else:
        node['left'] = best_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    if len(right) <= min_size:
        node['right'] = terminal_node(right)
    else:
        node['right'] = best_split(right)
        split(node['right'], max_depth, min_size, depth+1)


# Build a decision tree
def build_tree(training, max_depth, min_size):
    root = best_split(training)
    split(root, max_depth, min_size, 1)
    return root


def decision_tree(training, test, max_depth, min_size):
    tree = build_tree(training, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return(predictions)


def evaluate_model(train_set, test_set, decision_tree, max_depth, min_size):
    predicted_value = decision_tree(train_set, test_set, max_depth, min_size)
    actual = [row[-1] for row in test_set]
    metric = mean_squared_error(actual, predicted_value)
    return metric


# Print a decision tree
def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node)))



if __name__ == '__main__':
    s = time.time()
    dataset, test_set = importdata()
    max_depth = 4
    min_size = 10
    metric = evaluate_model(dataset[:,:],test_set, decision_tree, max_depth, min_size)
    print('Score: %s' % metric)
    metric_1 = evaluate_model(dataset[:, :], dataset[:,:], decision_tree, max_depth, min_size)
    print('training Score: %s' % metric_1)
    e = time.time()
    print("time",e-s)

