
import numpy as np
import pandas as pd
import time
from random import seed, randrange, shuffle
from multiprocessing import Process, Pipe

# Function importing Dataset
def importdata():
    spam_data = pd.read_csv(
        'http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data',
        sep=',', header=None)
    #for missing values remove row or column with (atleast 1) any Nan Null present
    # spam_data = spam_data.dropna(axis=0, how='any')
    return spam_data.values


# Calculate accuracy
def accuracy_val(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


#remove test set from training set
def removearray(train,arr):
    ind = 0
    size = len(train)
    while ind != size and not np.array_equal(train[ind],arr):
        ind += 1
    if ind != size:
        train.pop(ind)
    else:
        raise ValueError('not found')


def split_rows(index, threshold, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < threshold:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calculate the Gini index for a split dataset
def gini_index(groups, y):
    gini = 0.0
    total_group_size = float(sum([len(group) for group in groups]))
    for group in groups:
        size = float(len(group))
        val = 0.0
        # for divide by zero
        if size == 0:
            continue
        for y_val in y:
            p = [row[-1] for row in group].count(y_val) / size
            val += p * p
        # weight the group value by its relative size
        gini += (1.0 - val) * (size / total_group_size)
    return gini

def entropy(y):
    res = 0.0
    val, counts = np.unique(y, return_counts=True)
    freqs = counts.astype('float')/len(y)
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res


def info_gain(groups, y):
    left = np.array(groups[0])
    right = np.array(groups[1])
    y = np.array(y)
    old_entropy = entropy(y)
    ig = 0.0
    freqs = len(left)/len(y),len(right)/len(y)
    y_left = [row[-1] for row in left]
    y_right = [row[-1] for row in right]
    pl = freqs[0]
    pr = freqs [1]
    new_entropy = pl * entropy(y_left) + pr * entropy(y_right)
    ig = old_entropy - new_entropy
    print(ig)
    return ig


# Select the best split point for a dataset
def best_split(dataset):
    dataset = np.array(dataset)
    y_values = list(set(row[-1] for row in dataset))
    split_index, split_val, split_score, split_nodes = 99999999, 99999999, 99999999, None
    for index in range(len(dataset[0])-1):
        # sort by column
        dataset = dataset[dataset[:,index].argsort(kind='mergesort')]
        # feature = dataset[:,index]
        # max_fval = np.max(feature)
        # min_fval = np.min(feature)
        # feature = np.linspace(min_fval,max_fval,num=90)
        # feature = np.unique(feature)
        previous = list()
        for row in dataset:
            if row[index] not in previous:
                previous.append(row[index])
                groups = split_rows(index, row[index], dataset)
                gini = gini_index(groups, y_values)
                if gini < split_score:
                    split_index, split_val, split_score, split_nodes = index, row[index], gini, groups
    print('X%d < %.3f' % (split_index, split_val))
    return {'index':split_index, 'value':split_val, 'groups':split_nodes}


# def best_split(dataset):
#     dataset = np.array(dataset)
#     y_values = [row[-1] for row in dataset]
#     ig = 0.0
#     split_index, split_val, split_score, split_nodes = 0, 0.0, 0.0, None
#     for index in range(len(dataset[0])-1):
#         dataset = dataset[dataset[:, index].argsort(kind='mergesort')]
#         previous = list()
#         for row in dataset:
#             if row[index] not in previous:
#                 groups = split_rows(index, row[index], dataset)
#                 ig = info_gain(groups, y_values)
#                 if ig > split_score:
#                     split_index, split_val, split_score, split_nodes = index, row[index], ig, groups
#     print('Split: [X%d < %.3f] gain:%.3f' % (split_index, split_val, ig))
#     return {'index':split_index, 'value':split_val, 'groups':split_nodes}

# Create a terminal node value
def terminal_node(node):
    outcomes = [row[-1] for row in node]
    return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_node_size, depth):
    left, right = node['groups']
    del(node['groups'])
    # check for if left or right node is empty
    if not left or not right:
        node['left'] = node['right'] = terminal_node(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = terminal_node(left), terminal_node(right)
        return
    if len(left) <= min_node_size:
        node['left'] = terminal_node(left)
    else:
        node['left'] = best_split(left)
        split(node['left'], max_depth, min_node_size, depth+1)
    if len(right) <= min_node_size:
        node['right'] = terminal_node(right)
    else:
        node['right'] = best_split(right)
        split(node['right'], max_depth, min_node_size, depth+1)

# Build a decision tree
def build_tree(train, max_depth, min_node_size):
    print("training....")
    root = best_split(train)
    split(root, max_depth, min_node_size, 1)
    return root

# Print a decision tree
def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node)))

# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# Classification Tree Algorithm
def decision_tree(train, test, max_depth, min_node_size):
    tree = build_tree(train, max_depth, min_node_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return(predictions)

# Split a dataset into k folds
def k_fold_split(dataset, k_folds):
    dataset_split = list()
    dataset_1 = list(dataset)
    fold_size = int(len(dataset) / k_folds)
    for i in range(k_folds):
        fold = list()
        while len(fold) < fold_size:
            idx = randrange(len(dataset_1))
            fold.append(dataset_1.pop(idx))
        dataset_split.append(fold)
    return dataset_split


def evaluate_model(dataset, decision_tree, k_folds, max_depth, min_node_size):
    folds = k_fold_split(dataset, k_folds)
    scores = list()
    processes = list()
    connects = list()
    for fold in folds:
        parent_conn, child_conn = Pipe()
        p = Process(target=run_model,args=(fold,folds,decision_tree,max_depth,min_node_size,child_conn))
        p.start()
        connects.append(parent_conn)
        processes.append(p)
    for i,p in enumerate(processes):
        score = connects[i].recv()
        print("score", score)
        scores.append(score)
        p.join()
    return scores


def run_model(fold,folds,decision_tree,max_depth,min_node_size,conn):
    test_set = list()
    train_set = list(folds)
    removearray(train_set, fold)
    train_set = sum(train_set, [])
    for r in fold:
        row = list(r)
        test_set.append(row)
        row[-1] = None
    predicted = decision_tree(train_set, test_set, max_depth, min_node_size)
    actual = [r[-1] for r in fold]
    accuracy = accuracy_val(actual, predicted)
    conn.send(accuracy)
    conn.close()
    return accuracy




if __name__ == '__main__':
    s = time.time()
    seed(1)
    dataset = importdata()
    # shuffle(dataset)
    k_folds = 5
    max_depth = 5
    min_node_size = 10
    accuracy = evaluate_model(dataset, decision_tree, k_folds, max_depth, min_node_size)
    print('Accuracy: %s' % accuracy)
    print('Mean Accuracy: %.3f%%' % (round(sum(accuracy) / float(len(accuracy)))))
    e = time.time()
    print("time", e - s)
