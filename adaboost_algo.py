from __future__ import division, print_function
import numpy as np
import math
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics  import accuracy_score
from sklearn import metrics


#Helper Function
# Function importing Dataset
def importdata(file):
    spam_data = pd.read_csv(file, sep=' ', header=None)
    #for missing values remove row or column with (atleast 1) any Nan Null present
    # spam_data = spam_data.dropna(axis=0, how='any')
    print(np.shape(spam_data))
    return spam_data.values


# Function to split the dataset
def splitdataset(data):
    # Seperating the target variable
    x = data[:, 0:data.shape[1]-1]
    y = data[:, data.shape[1]-1]
    print(np.shape(x),np.shape(y))
    return x, y


# Decision stump used as weak classifier in this impl. of Adaboost
class DecisionStump():
    def __init__(self):
        # indicate whether sample shall be classified as -1 or 1 given threshold
        self.sign_polarity = 1
        self.feature_index = None
        self.threshold = None
        # Value indicative of the classifier's accuracy
        self.alpha = None


class Adaboost():

    # Parameter T: # of rounds / weak classifiers
    def __init__(self, T):
        self.T = T

    def fit(self, X, y,X_test, y_test):
        n_samples, n_features = np.shape(X)
        # Initialize weights to 1/M
        w = np.full(n_samples, (1 / n_samples))
        self.clfs = []  # save all the best classfier (with min error)

        # Iterate through classifiers / rounds
        for t in range(self.T):
            clf = DecisionStump()
            # Minimum error given for using a certain (feature, threshold) for predicting sample label
            min_error = float('inf')
            # Iterate throught every unique feature value and see what value makes the best threshold for predicting y
            # running with optimal decision stumps
            for feature_i in range(n_features):
                features_values = X[:, feature_i]
                unique_values = list(set(features_values))
                idx = np.linspace(0, len(unique_values)-1, 5).astype('int') #interval approach
                thresholds = [unique_values[i] for i in idx]
                for threshold in thresholds:
                    p = 1
                    # Set all predictions to '1' initially
                    prediction = np.ones(np.shape(y))
                    # Label the samples whose values are below threshold as '-1'
                    prediction[X[:, feature_i] < threshold] = -1
                    # Error = sum of weights of misclassified samples
                    error = sum(w[y != prediction])
                    # If the error is over 50% we flip the sign_polarity so that samples that  were classified as 0 are classified as 1, and vice versa
                    if error > 0.5:
                        error = 1 - error
                        p = -1
                    # If this threshold resulted in the smallest error we save the stumps
                    if error < min_error:
                        clf.sign_polarity = p
                        clf.threshold = threshold
                        clf.feature_index = feature_i
                        min_error = error

            #==================== USING RANDOM STUMPS ===============================
            # feature_i = np.random.choice(list(range(n_features)))
            # feature_values = X[:, feature_i]
            # # unique_values = np.unique(feature_values)
            # unique_values = list(set(feature_values))
            # threshold = np.random.choice(unique_values)
            # p = 1
            # # Set all predictions to '1' initially
            # prediction = np.ones(np.shape(y))
            # # Label the samples whose values are below threshold as '-1'
            # prediction[X[:, feature_i] <  threshold] = -1
            # # Error = sum of weights of misclassified samples
            # error = sum(w[y != prediction])
            # if error > 0.5:
            #     error = 1 - error
            #     p = -1
            # clf.sign_polarity = p
            # clf.threshold = threshold
            # clf.feature_index = feature_i
            # min_error = error
             # ===================================================
            # Calculate the alpha which is used to update the sample weights,
            clf.alpha = 0.5 * math.log((1.0 - min_error) / (min_error + 1e-10))
            # Set all predictions to '1' initially
            predictions = np.ones(np.shape(y))
            # The indexes where the sample values are below threshold
            negative_idx = (clf.sign_polarity * X[:, clf.feature_index] < clf.sign_polarity * clf.threshold)
            # Label those as '-1'
            predictions[negative_idx] = -1
            # Calculate new weights
            # Missclassified samples gets larger weights and correctly classified samples smaller
            w *= np.exp(-clf.alpha * y * predictions)
            # Normalize to one
            w /= np.sum(w)

            y_pred = self.predict_all(X)
            train_error = sum(y_pred != y) / float(len(y))

            y_pred_test = self.predict_all(X_test)
            test_error = sum(y_pred_test != y_test) / float(len(y_test))

            print("rounds ", t + 1, " round_err ", min_error, " fi ", clf.feature_index, " th ", clf.threshold,
                  " train_err ", train_error, " test_error", test_error)
            # add the classifier to list
            self.clfs.append(clf)



    def predict_all(self, X):
        n_samples = np.shape(X)[0]
        y_pred = np.zeros((n_samples, 1))
        # For each classifier => label the samples
        for clf in self.clfs:
            # Set all predictions to '1' initially
            predictions = np.ones(np.shape(y_pred))
            # The indexes where the sample values are below threshold
            negative_idx = (clf.sign_polarity * X[:, clf.feature_index] < clf.sign_polarity * clf.threshold)
            # Label those as '-1'
            predictions[negative_idx] = -1
            # Add predictions weighted by the classifiers alpha
            y_pred += clf.alpha * predictions

        # Return sign of prediction sum
        y_pred = np.sign(y_pred).flatten()
        return y_pred

    def feature_extract(self):
        features = []
        for clf in self.clfs:
            features.append(clf.feature_index)
        return features[:10]



def main():
    start = time.time()
    X_train = importdata('spam_polluted/train_feature.txt')
    X_test = importdata('spam_polluted/test_feature.txt')
    y_train = importdata('spam_polluted/train_label.txt')
    y_test = importdata('spam_polluted/test_label.txt')

    y_train = y_train.reshape((np.shape(y_train)[0]))
    print(np.shape(y_train))
    y_test = y_test.reshape((np.shape(y_test)[0]))
    print(np.shape(y_test))

    label1 = 0
    label2 = 1
    idx = np.append(np.where(y_train == label1)[0], np.where(y_train == label2)[0])
    y_train = y_train[idx]
    # Changing labels to {-1, 1}
    y_train[y_train == label1] = -1
    y_train[y_train == label2] = 1
    X_train = X_train[idx]  #sorting the dataset in order of classes

    # Changing labels to {-1, 1}
    y_test[y_test == label1] = -1
    y_test[y_test == label2] = 1

    # Adaboost classification with 100 weak classifiers
    clf = Adaboost(T=300)
    clf.fit(X_train, y_train,X_test,y_test)

    y_pred = clf.predict_all(X_test)
    # print(y_pred)
    # test_error = sum(y_pred != y_test) / float(len(y_test))

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # features = clf.feature_extract()
    # print(features)

    end = time.time()
    print("time", end - start)





if __name__ == "__main__":
    main()
