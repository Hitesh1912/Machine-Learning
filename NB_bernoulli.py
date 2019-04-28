#AUTHOR : HITESH VERMA

import time
from random import randrange

from tricks.utilities import *


# Function importing Dataset
def importdata():
    # data = pd.read_csv(
    #     '/MSCS/ML/code/HW2/spambase_csv.csv', sep=',', header=None)
    data = pd.read_csv(
        'http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data',
        sep=',', header=None)
    # #for missing values remove row or column with (atleast 1) any Nan Null present
    # data = data.dropna(axis=0, how='any')
    print(data.shape)
    return data.values

# Function to split the dataset
def splitdataset(data):
    # Seperating the target variable
    x = data[:, 0:data.shape[1]-1]
    y = data[:, data.shape[1]-1]
    return x, y


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


# Calculate accuracy
def accuracy_val(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


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


def seperated_by_class(X,y):
    group = {}
    class1_indices = np.where(y == 0)[0]
    class2_indices = np.where(y == 1)[0]
    group[0] = X[class1_indices]
    group[1] = X[class2_indices]
    return group


#calculate overalll mean of each feature
def cal_mean_per_feature(X):
    mean_per_feature = {}
    n = np.shape(X)[1]
    for i in range(n):
        mean_per_feature[i] = np.mean(X[:,i])
    # print("mean table",mean_per_feature)
    return mean_per_feature


#calculate likelihood (probability) per feature for a class
def cal_feature_distribution(group_by_class, feature_mean):
    n = np.shape(group_by_class)[1]
    m = np.shape(group_by_class)[0]
    print("checker",m)
    k = 2 #should be variable
    feature_model = {}
    count_less_than_mu = 0
    for i in range(n):
        feature_col = group_by_class[:,i]
        count_less_than_mu = len([x for x in feature_col if x <= feature_mean[i]])
        count_more_than_mu = len([x for x in feature_col if x > feature_mean[i]])
        #using laplace smoothing
        freq1 = (count_less_than_mu + 1) / (m + k)       # below mu
        # freq2 = ((m - count_less_than_mu) + 1) / (m + k) # above mu
        freq2 = (count_more_than_mu + 1) / (m + k)
        feature_model[i] = {0.:freq1 ,1.:freq2}
    # print(feature_model)
    return feature_model


#return likelihood (feature propabalities) per feature per class
def estimate_parameter(X,y,feature_mean):
    group = seperated_by_class(X, y)
    feature_estimate = {}
    for class_label, x_group in group.items():
        feature_estimate[class_label] = cal_feature_distribution(x_group,feature_mean)
    # print(feature_estimate)
    return feature_estimate


#likelihood
#calculate class prob (0 or 1) for each data point
def cal_class_prob(test_row, prior, feature_estimate, feature_mean):
    results = {}
    for class_val, feature_prob in feature_estimate.items():
        class_probability = prior[class_val]
        for i in range(0, len(test_row)):
            relative_feature_values = feature_prob[i]
            #if test_row[i] in elative_feature_values.keys():
            x = 0.0 if test_row[i] <= feature_mean[i] else 1.0
            # if x in relative_feature_values.keys()
            class_probability *= relative_feature_values[x]  #bug
        results[class_val] = class_probability
    # print(results)
    return results


#predict class for each data point
def b_predict(x, prior, feature_estimate,feature_mean):
    class_probs = cal_class_prob(x, prior, feature_estimate,feature_mean)
    bestLabel, bestProb = None, -1
    for class_value, probability in class_probs.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = class_value
    return bestLabel,class_probs


def b_getPredictions(test_data, prior_y,feature_estimate,feature_mean):
    predictions = []
    prob = []
    for i in range(len(test_data)):
        result,class_probs = b_predict(test_data[i], prior_y, feature_estimate,feature_mean)
        predictions.append(result)
        prob.append(class_probs)
    return predictions, prob

def prior(X,y):
    m = np.shape(y)[0]
    p_y = {}
    group = seperated_by_class(X, y)
    for class_val, x_class in group.items():
        count_spam = len(x_class)
        p_y[class_val] = count_spam / m
    # print("p_y", p_y)
    return p_y


def evaluate_model(dataset, k_folds):
    folds = k_fold_split(dataset, k_folds)
    test_scores = list()
    train_scores = list()
    print("training model.....")
    fold_count = 0
    for fold in folds:
        fold_count += 1
        test_set = list()
        train_set = list(folds)
        removearray(train_set, fold)
        train_set = sum(train_set, [])
        for r in fold:
            row = list(r)
            test_set.append(row)
        train_set = np.array(train_set)
        test_set = np.array(test_set)

        #splitting dataset into features and labels
        X, y = splitdataset(train_set)
        X_test, y_test = splitdataset(test_set)

        feature_mean = cal_mean_per_feature(X)
        # bernoulli (boolean) distribution
        feature_estimate = estimate_parameter(X, y, feature_mean)
        # print(feature_estimate)
        p_y = prior(X, y)

        # ===============running model on train set=====================#

        # predictions = b_getPredictions(X, p_y, feature_estimate,feature_mean)
        # accuracy = accuracy_val(y, predictions)
        # train_scores.append(accuracy)
    #
        #===============running model  on test set=====================#

        predictions, prob_scores =  b_getPredictions(X_test, p_y,feature_estimate,feature_mean)
        # print("predictions",prob_scores)
        if fold_count == 1:
            plot_roc(prob_scores, predictions)
            # display confusion matrix
            confusion_matrix(y_test, predictions)

        accuracy = accuracy_val(y_test, predictions)
        test_scores.append(accuracy)

    # print("train_accuracy", train_scores)
    print("test_accuracy",test_scores)
    print("testing of model complete")
    return test_scores, train_scores




if __name__ == '__main__':
    seed = 1
    s = time.time()
    dataset = importdata()
    # shuffle(dataset)
    test_acc, train_acc = evaluate_model(dataset, 2)
    print('Test Accuracy: %s' % test_acc)
    print('Test Acc: %.3f%%' % (sum(test_acc) / float(len(test_acc))))
    # print('Train accuracy: %s' % train_acc)
    # print('Train Acc: %.3f%%' % (sum(train_acc) / float(len(train_acc))))
    e = time.time()
    print("time",e-s)