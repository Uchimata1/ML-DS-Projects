import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def euclidean_dist(x1, x2):
    dist = np.sqrt(np.sum((x1-x2)**2))
    return dist

class Knn(object):
    k = 0    # number of neighbors to use

    def __init__(self, k):
        """
        Knn constructor

        Parameters
        ----------
        k : int 
            Number of neighbors to use.
        """
        self.k = k

    def train(self, xFeat, y):
        """
        Train the k-nn model.

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """
        xFeat = xFeat.to_numpy()

        self.X_train = xFeat
        self.y_train = y
        return self


    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.  

        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted class label per sample
        """
        xFeat = xFeat.to_numpy() # convert to numpy array

        yHat = [self._predict(x) for x in xFeat] # variable to store the estimated class label
        yHat = np.array(yHat)
        # TODO
        return yHat
    
    def _predict(self, x): 
        # compute distances
        distances = [euclidean_dist(x, x_train) for x_train in self.X_train]
        
        # get the closest k
        indices = np.argsort(distances)[:self.k]
        kn_labels = [self.y_train[i] for i in indices]

        # majority vote 
        num_zeros = 0
        num_ones = 0
        for label in kn_labels: 
            if (label == 0): 
                num_zeros += 1
            else: 
                num_ones += 1
        if (num_zeros > num_ones): 
            return 0
        else: 
            return 1

def accuracy(yHat, yTrue):
    """
    Calculate the accuracy of the prediction

    Parameters
    ----------
    yHat : 1d-array with shape n
        Predicted class label for n samples
    yTrue : 1d-array with shape n
        True labels associated with the n samples

    Returns
    -------
    acc : float between [0,1]
        The accuracy of the model
    """
    # calculate the accuracy
    num_correct = 0
    for i in range(len(yHat)):
        if (yHat[i] == yTrue[i]):
            num_correct += 1

    acc = num_correct / len(yHat) 
    return acc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    parser.add_argument("--xTrain",
                        default="q3xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q3yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q3xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q3yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    # create an instance of the model
    knn = Knn(args.k)
    knn.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = knn.predict(xTrain)
    trainAcc = accuracy(yHatTrain, yTrain['label'])
    # predict the test dataset
    yHatTest = knn.predict(xTest)
    testAcc = accuracy(yHatTest, yTest['label'])
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)

    # 3: What is the Time Complexity 
    # Time Complexity: O(d*n^2) where n = # of rows. The predict function performs a nested for-loop 
    # n times each thus that results in n^2. The euclidean_dist is 0(d) because d operations are made in 
    # np.sum(). That results in O(d*n^2). k doesn't affect the time complexity. 

    # collect data for plotting of accuracies 
    ks = [1, 3, 5, 7, 9, 11, 13, 15]
    train_acc = []
    test_acc = []
    for k in ks: 
        knn = Knn(k)
        knn.train(xTrain, yTrain['label'])
        # predict the training dataset
        yHatTrain = knn.predict(xTrain)
        trainAcc = accuracy(yHatTrain, yTrain['label'])
        train_acc.append(trainAcc)
        # predict the test dataset
        yHatTest = knn.predict(xTest)
        testAcc = accuracy(yHatTest, yTest['label'])
        test_acc.append(testAcc)
    
    sns.lineplot(x = ks, y = train_acc, label = "Training Accuracy")
    sns.lineplot(x = ks, y = test_acc, label = "Testing Accuracy").set(xlabel = "k", ylabel = "accuracy")
    plt.show()



if __name__ == "__main__":
    main()