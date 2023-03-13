import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
import time

 
def holdout(model, xFeat, y, testSize):
    """
    Split xFeat into random train and test based on the testSize and
    return the model performance on the training and test set. 

    Parameters
    ----------
    model : sktree.DecisionTreeClassifier
        Decision tree model
    xFeat : nd-array with shape n x d
        Features of the dataset 
    y : 1-array with shape n x 1
        Labels of the dataset
    testSize : float
        Portion of the dataset to serve as a holdout. 

    Returns
    -------
    trainAuc : float
        Average AUC of the model on the training dataset
    testAuc : float
        Average AUC of the model on the validation dataset
    timeElapsed: float
        Time it took to run this function
    """
    ## split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(xFeat, y, random_state = 42, test_size=testSize)

    ## create decision tree
    dt = model.fit(X_train, Y_train)

    ## make prediction
    prediction_xTrain = dt.predict_proba(X_train)
    prediction_xTest = dt.predict_proba(X_test)

    ## input score
    start = time.time() # start time
    trainAuc = metrics.roc_auc_score(Y_train, prediction_xTrain[:, 1])
    testAuc = metrics.roc_auc_score(Y_test, prediction_xTest[:, 1])
    end = time.time() # end time 
    timeElapsed = end - start

    return trainAuc, testAuc, timeElapsed


def kfold_cv(model, xFeat, y, k):
    """
    Split xFeat into k different groups, and then use each of the
    k-folds as a validation set, with the model fitting on the remaining
    k-1 folds. Return the model performance on the training and
    validation (test) set. 


    Parameters
    ----------
    model : sktree.DecisionTreeClassifier
        Decision tree model
    xFeat : nd-array with shape n x d
        Features of the dataset 
    y : 1-array with shape n x 1
        Labels of the dataset
    k : int
        Number of folds or groups (approximately equal size)

    Returns
    -------
    trainAuc : float
        Average AUC of the model on the training dataset
    testAuc : float
        Average AUC of the model on the validation dataset
    timeElapsed: float
        Time it took to run this function
    """
    start = time.time()
    xFeat = xFeat.to_numpy()
    y = y.to_numpy()
    kf = KFold(n_splits=k)
    trainAuc_s, testAuc_s = [], []

    for trainIndex, testIndex in kf.split(xFeat):
        X_train, X_test = xFeat[trainIndex], xFeat[testIndex]
        y_train, y_test = y[trainIndex], y[testIndex]

        # train and evaluate 
        dt_train = model.fit(X_train, y_train)

        # make prediction 
        prediction_xTrain = dt_train.predict_proba(X_train)
        prediction_xTest = dt_train.predict_proba(X_test)

        # calculate AUC
        trainAuc = metrics.roc_auc_score(y_train, prediction_xTrain[:, 1])
        testAuc = metrics.roc_auc_score(y_test, prediction_xTest[:, 1])

        # append to array
        trainAuc_s.append(trainAuc)
        testAuc_s.append(testAuc)

    trainAuc = np.mean(np.array(trainAuc_s))
    testAuc = np.mean(np.array(testAuc_s))

    end = time.time()
    timeElapsed = end - start

    return trainAuc, testAuc, timeElapsed


def mc_cv(model, xFeat, y, testSize, s):
    """
    Evaluate the model using s samples from the
    Monte Carlo cross validation approach where
    for each sample you split xFeat into
    random train and test based on the testSize.
    Returns the model performance on the training and
    test datasets.

    Parameters
    ----------
    model : sktree.DecisionTreeClassifier
        Decision tree model
    xFeat : nd-array with shape n x d
        Features of the dataset 
    y : 1-array with shape n x 1
        Labels of the dataset
    testSize : float
        Portion of the dataset to serve as a holdout. 

    Returns
    -------
    trainAuc : float
        Average AUC of the model on the training dataset
    testAuc : float
        Average AUC of the model on the validation dataset
    timeElapsed: float
        Time it took to run this function
    """
    start = time.time()
    xFeat = xFeat.to_numpy()
    y = y.to_numpy()
    trainAuc_s, testAuc_s = [], []

    ss = ShuffleSplit(n_splits=s, test_size = testSize)

    for trainIndex, testIndex in ss.split(xFeat, y):
        X_train, X_test = xFeat[trainIndex], xFeat[testIndex]
        y_train, y_test = y[trainIndex], y[testIndex]

        # train and evaluate 
        dt_train = model.fit(X_train, y_train)

        # make prediction 
        prediction_xTrain = dt_train.predict_proba(X_train)
        prediction_xTest = dt_train.predict_proba(X_test)

        # calculate AUC
        trainAuc = metrics.roc_auc_score(y_train, prediction_xTrain[:, 1])
        testAuc = metrics.roc_auc_score(y_test, prediction_xTest[:, 1])

        # append to array
        trainAuc_s.append(trainAuc)
        testAuc_s.append(testAuc)

    trainAuc = np.mean(np.array(trainAuc_s))
    testAuc = np.mean(np.array(testAuc_s))

    end = time.time()
    timeElapsed = end - start

    return trainAuc, testAuc, timeElapsed


def sktree_train_test(model, xTrain, yTrain, xTest, yTest):
    """
    Given a sklearn tree model, train the model using
    the training dataset, and evaluate the model on the
    test dataset.

    Parameters
    ----------
    model : DecisionTreeClassifier object
        An instance of the decision tree classifier 
    xTrain : nd-array with shape nxd
        Training data
    yTrain : 1d array with shape n
        Array of labels associated with training data
    xTest : nd-array with shape mxd
        Test data
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    trainAUC : float
        The AUC of the model evaluated on the training data.
    testAuc : float
        The AUC of the model evaluated on the test data.
    """
    # fit the data to the training dataset
    model.fit(xTrain, yTrain)
    # predict training and testing probabilties
    yHatTrain = model.predict_proba(xTrain)
    yHatTest = model.predict_proba(xTest)
    # calculate auc for training
    fpr, tpr, thresholds = metrics.roc_curve(yTrain['label'],
                                             yHatTrain[:, 1])
    trainAuc = metrics.auc(fpr, tpr)
    # calculate auc for test dataset
    fpr, tpr, thresholds = metrics.roc_curve(yTest['label'],
                                             yHatTest[:, 1])
    testAuc = metrics.auc(fpr, tpr)
    return trainAuc, testAuc


def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
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
    # create the decision tree classifier
    dtClass = DecisionTreeClassifier(max_depth=15,
                                     min_samples_leaf=10)
    # use the holdout set with a validation size of 30 of training
    aucTrain1, aucVal1, time1 = holdout(dtClass, xTrain, yTrain, 0.70)
    # use 2-fold validation
    aucTrain2, aucVal2, time2 = kfold_cv(dtClass, xTrain, yTrain, 2)
    # use 5-fold validation
    aucTrain3, aucVal3, time3 = kfold_cv(dtClass, xTrain, yTrain, 5)
    # use 10-fold validation
    aucTrain4, aucVal4, time4 = kfold_cv(dtClass, xTrain, yTrain, 10)
    # use MCCV with 5 samples
    aucTrain5, aucVal5, time5 = mc_cv(dtClass, xTrain, yTrain, 0.70, 5)
    # use MCCV with 10 samples
    aucTrain6, aucVal6, time6 = mc_cv(dtClass, xTrain, yTrain, 0.70, 10)
    # train it using all the data and assess the true value
    trainAuc, testAuc = sktree_train_test(dtClass, xTrain, yTrain, xTest, yTest)
    perfDF = pd.DataFrame([['Holdout', aucTrain1, aucVal1, time1],
                           ['2-fold', aucTrain2, aucVal2, time2],
                           ['5-fold', aucTrain3, aucVal3, time3],
                           ['10-fold', aucTrain4, aucVal4, time4],
                           ['MCCV w/ 5', aucTrain5, aucVal5, time5],
                           ['MCCV w/ 10', aucTrain6, aucVal6, time6],
                           ['True Test', trainAuc, testAuc, 0]],
                           columns=['Strategy', 'TrainAUC', 'ValAUC', 'Time'])
    print(perfDF)


    # Problem 2d: 
        # Regards to AUC: all the model selection techniques are very similar. 
        # All produce roughly 93-95% AUC. However Holdout is on the lower end

        # Regards to robustness	of	the	validation	estimate: From the table 5-fold, 10-fold, MCCV w/ 5, MCCV w/ are 
        # the most robust robust because k-fold and Monte Carlo split up and compute the average AUC. 

        # Regards to computional time: 2-fold, and 5-fold appear to have the highest time. Then followed by 2-fold, 
        # Holdout, and than Monte Carlo w/ s = 5,10. 


if __name__ == "__main__":
    main()
