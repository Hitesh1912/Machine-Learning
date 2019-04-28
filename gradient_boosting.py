
from regression_tree import *


class GradientBoosting(object):

    def __init__(self, T):
        self.T = T


    def fit(self, training, max_depth, min_size):
        X, y = splitdataset(training)
        X = np.array(X)
        y = np.array(y)
        # Initialize regression trees
        self.trees = []
        for _ in range(self.T):
            tree = build_tree(training, max_depth, min_size)
            y_pred = self.y_predict(tree,training)
            y = y - y_pred
            y = np.expand_dims(y, axis=1)
            #print("checker",np.shape(X), np.shape(y))
            training = np.concatenate((X,y),axis=1)
            y = y.reshape(len(y), )
            #print(np.shape(training))
            self.trees.append(tree)

    def y_predict(self,tree, data):
        predictions = list()
        for row in data:
            prediction = predict(tree, row)
            predictions.append(prediction)
        return (predictions)

    def predict_all(self, test_data):
        n_samples = np.shape(test_data)[0]
        # y_pred = np.zeros((n_samples, 1))
        predictions = []
        # For each classifier => label the samples
        for tree in self.trees:
            y_pred = self.y_predict(tree,test_data)
            predictions.append(y_pred)
        predictions = np.array(predictions)
        print(np.shape(predictions))
        # sum of all the predictions of all the trees
        tot_pred = np.sum(predictions, axis=0)
        return tot_pred



def main():
    train_set, test_set = importdata()
    X_test, y_test = splitdataset(test_set)
    X_train, y_train = splitdataset(train_set)
    reg = GradientBoosting(T=10)
    max_depth = 2
    min_size = 10
    reg.fit(train_set, max_depth, min_size)

    tot_pred = reg.predict_all(test_set)
    print(tot_pred)
    # test_error = sum(tot_pred != y_test) / float(len(y_test))
    print("testing error", mean_squared_error(y_test, tot_pred))

    tot_train_pred = reg.predict_all(train_set)
    # train_error = sum(tot_train_pred != y_train) / float(len(y_train))
    print("training error", mean_squared_error(y_train, tot_train_pred))




if __name__ == '__main__':
    main()