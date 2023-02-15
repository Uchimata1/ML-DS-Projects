import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import knn


def standard_scale(xTrain, xTest):
    """
    Preprocess the training data to have zero mean and unit variance.
    The same transformation should be used on the test data. For example,
    if the mean and std deviation of feature 1 is 2 and 1.5, then each
    value of feature 1 in the test set is standardized using (x-2)/1.5.

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data 
    xTest : nd-array with shape m x d
        Test data 

    Returns
    -------
    xTrain : nd-array with shape n x d
        Transformed training data with mean 0 and unit variance 
    xTest : nd-array with shape m x d
        Transformed test data using same process as training.
    """
    xTrain = xTrain.copy()
    xTest = xTest.copy()

    # standarize xTrain
    j = -1
    for col in xTrain.columns:
        mean = np.mean(xTrain[col])
        sd = np.std(xTrain[col])
        j += 1
        for i in range(xTrain.shape[0]):
            xTrain.iloc[i, j] = (xTrain.iloc[i, j] - mean) / sd
    
    # standarize xTest
    j = -1
    for col in xTest.columns:
        mean = np.mean(xTest[col])
        sd = np.std(xTest[col])
        j += 1
        for i in range(xTest.shape[0]):
            xTest.iloc[i, j] = (xTest.iloc[i, j] - mean) / sd

    return xTrain, xTest


def minmax_range(xTrain, xTest):
    """
    Preprocess the data to have minimum value of 0 and maximum
    value of 1.T he same transformation should be used on the test data.
    For example, if the minimum and maximum of feature 1 is 0.5 and 2, then
    then feature 1 of test data is calculated as:
    (1 / (2 - 0.5)) * x - 0.5 * (1 / (2 - 0.5))

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data 
    xTest : nd-array with shape m x d
        Test data 

    Returns
    -------
    xTrain : nd-array with shape n x d
        Transformed training data with min 0 and max 1.
    xTest : nd-array with shape m x d
        Transformed test data using same process as training.
    """    
    xTrain = xTrain.copy()
    xTest = xTest.copy()

    # min-max normalize xTrain
    j = -1
    for col in xTrain.columns:
        min = np.min(xTrain[col])
        max = np.max(xTrain[col])
        j += 1
        for i in range(xTrain.shape[0]):
            x = xTrain.iloc[i, j]
            xTrain.iloc[i, j] = (x - min) / (max - min)
    
    # min-max normalize xTest
    j = -1
    for col in xTest.columns:
        min = np.min(xTest[col])
        max = np.max(xTest[col])
        j += 1
        for i in range(xTest.shape[0]):
            x = xTest.iloc[i, j]
            xTest.iloc[i, j] = (x - min) / (max - min)

    return xTrain, xTest


def add_irr_feature(xTrain, xTest):
    """
    Add 2 features using Gaussian distribution with 0 mean,
    standard deviation of 1.

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data 
    xTest : nd-array with shape m x d
        Test data 

    Returns
    -------
    xTrain : nd-array with shape n x (d+2)
        Training data with 2 new noisy Gaussian features
    xTest : nd-array with shape m x (d+2)
        Test data with 2 new noisy Gaussian features
    """
    xTrain = xTrain.copy()
    xTest = xTest.copy()

    # add two irr variables to xTrain
    xTrain['irr1'] = np.random.normal(0, 1, xTrain.shape[0])
    xTrain['irr2'] = np.random.normal(0, 1, xTrain.shape[0])

    # add two irr variables to xTest
    xTest['irr1'] = np.random.normal(0, 1, xTest.shape[0])
    xTest['irr2'] = np.random.normal(0, 1, xTest.shape[0])

    return xTrain, xTest


def knn_train_test(k, xTrain, yTrain, xTest, yTest):
    """
    Given a specified k, train the knn model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    k : int
        The number of neighbors
    xTrain : nd-array with shape n x d
        Training data 
    yTrain : 1d array with shape n
        Array of labels associated with training data.
    xTest : nd-array with shape m x d
        Test data 
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    acc : float
        The accuracy of the trained knn model on the test data
    """
    model = knn.Knn(k)
    model.train(xTrain, yTrain['label'])
    # predict the test dataset
    yHatTest = model.predict(xTest)
    return knn.accuracy(yHatTest, yTest['label'])
    

def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    parser.add_argument("--xTrain",
                        default="q4xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q4yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q4xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q4yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)

    # no preprocessing
    acc1 = knn_train_test(args.k, xTrain, yTrain, xTest, yTest)
    print("Test Acc (no-preprocessing):", acc1)
    # preprocess the data using standardization scaling
    xTrainStd, xTestStd = standard_scale(xTrain, xTest)
    acc2 = knn_train_test(args.k, xTrainStd, yTrain, xTestStd, yTest)
    print("Test Acc (standard scale):", acc2)
    # preprocess the data using min max scaling
    xTrainMM, xTestMM = minmax_range(xTrain, xTest)
    acc3 = knn_train_test(args.k, xTrainMM, yTrain, xTestMM, yTest)
    print("Test Acc (min max scale):", acc3)
    # add irrelevant features
    xTrainIrr, yTrainIrr = add_irr_feature(xTrain, xTest)
    acc4 = knn_train_test(args.k, xTrainIrr, yTrain, yTrainIrr, yTest)
    print("Test Acc (with irrelevant feature):", acc4)

    # collect data for plotting of accuracies 
    ks = [1, 3, 5, 7, 9, 11, 13, 15]
    znorm = []
    minmax = []
    znorm_noise = []
    minmax_noise = []
    for k in ks: 
        # w/o irr features
        acc1 = knn_train_test(k, xTrainStd, yTrain, xTestStd, yTest) # Std
        znorm.append(acc1)
        
        acc2 = knn_train_test(k, xTrainMM, yTrain, xTestMM, yTest) # min-max
        minmax.append(acc2)

        # w/ irr features
        xTrainStd_irr, xTestStd_irr = standard_scale(xTrainIrr, yTrainIrr)
        xTrainMM_irr, xTestMM_irr = minmax_range(xTrainIrr, yTrainIrr)
        acc3 = knn_train_test(k, xTrainStd_irr, yTrain, xTestStd_irr, yTest) # Std w/ noise
        acc4 = knn_train_test(k, xTrainMM_irr, yTrain, xTestMM_irr, yTest) # min-max w/ noise
        znorm_noise.append(acc3)
        minmax_noise.append(acc4)
    
    sns.lineplot(x = ks, y = znorm, label = "Test_acc for Z-Score Norm.")
    sns.lineplot(x = ks, y = znorm_noise, label = "Test_acc for Z-score Norm. w/ Noise")
    sns.lineplot(x = ks, y = minmax, label = "Test_acc for Min-Max Norm.")
    sns.lineplot(x = ks, y = minmax_noise, label = "Test_acc for Min-Max Norm. w/ Noise").set(
        xlabel = "k", ylabel = "accuracy")
    plt.show()  

    # 4d: different	forms of preprocessing and the sensitivity to irrelevant feature for this dataset?  
    # From the graph it appears that z-score norm. handles irrelevant features better in terms of accuracy
    

if __name__ == "__main__":
    main()
