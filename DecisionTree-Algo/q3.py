import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import time

 
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
    y = np.array(y)
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
    

    # Problem 3a Part 1: KNN
    # I chose k = 5 because from q2.py it had the highest valAuc. So it is reasonable to assume that k = 5 is best. 
    ks = [x for x in range(1, 20)]
    k_loop_values = []

    for k in ks:  # Run 5 fold cross validation
        knn_model = KNeighborsClassifier(n_neighbors = k)
        yTrain = np.ravel(yTrain)
        trainAuc, testAuc, timeElapsed = kfold_cv(knn_model, xTrain, yTrain, 5)
        k_loop_values.append([k, testAuc])

    k_loop_values = np.array(k_loop_values)
    max_AUC = np.argmax(k_loop_values[:, 1])
    optimal = k_loop_values[max_AUC]

    print("KNeighborsClassifier:")
    print("Optimal K-value =", optimal[0])

    
    # Problem 3a Part 2: DecisionTree
    depth = [x for x in range(2, 30)]
    min_samples_leaf = [x for x in range(1, 30)]
    params = []

    for dep in depth:  # Run 5 fold cross validation
        for minleaf in min_samples_leaf: 
            dt_model = DecisionTreeClassifier(max_depth=dep, min_samples_leaf=minleaf)
            trainAuc, testAuc, timeElapsed = kfold_cv(dt_model, xTrain, yTrain, 5)
            params.append([dep, minleaf, testAuc])
    

    np_params = np.array(params)
    print("max AUC: ", np.max(np_params[:, -1]))
    max_AUC = np.argmax(np_params[:, -1])
    optimal_params = np_params[max_AUC]
    
    print("DecisionTreeClassifier:")
    print("[depth, min_samples_leaf, testAuc] =", optimal_params)
    print("Optimal depth =", optimal_params[0])
    print("Optimal min_samples_leaf =", optimal_params[1])

    # Problem 3b and 3c
    perct = [0, 0.05, .10, 0.20]
    knn_array = []
    dt_array = []
    for per in perct:
        # remove rows 
        optimal_k_value = 14 #optimal[0]
        nSamples = int(xTrain.shape[0]*(1 - per))
        idx = np.random.choice(xTrain.shape[0], nSamples, replace = False)
        xTrainSub = xTrain.iloc[idx, :]
        yTrainSub = yTrain[idx]

        # 3b: model and predictions for knn
        knn_model = KNeighborsClassifier(n_neighbors = optimal_k_value)
        knn = knn_model.fit(xTrainSub, yTrainSub)
        y_predict = knn.predict(xTest)
        accuracy = metrics.accuracy_score(yTest, y_predict)
        prediction_xTest = knn.predict_proba(xTest)
        testAuc = metrics.roc_auc_score(yTest, prediction_xTest[:, 1])
        knn_array.append([accuracy, testAuc])

        # 3c: model and predictions for decision tree
        dt = DecisionTreeClassifier(max_depth = int(optimal_params[0]), min_samples_leaf = int(optimal_params[1]))
        dt = dt.fit(xTrainSub, yTrainSub)
        y_predict = dt.predict(xTest)
        accuracy = metrics.accuracy_score(yTest, y_predict)
        prediction_xTest = knn.predict_proba(xTest)
        testAuc = metrics.roc_auc_score(yTest, prediction_xTest[:, 1])
        dt_array.append([accuracy, testAuc])


    # Problem 3d: table 
    knn_array = np.array(knn_array)
    dt_array = np.array(dt_array)
    data_knn = {'knn_accuracy': knn_array[:, 0], 'knn_testAuc': knn_array[:, 1], 'dt_accuracy': dt_array[:, 0], 'dt_testAuc': dt_array[:, 1]}
    knn_table = pd.DataFrame(data_knn, index = ["0%", "5%", "10%", "20%"])
    print(knn_table)

    # they are not that sensitivite. The accuracy and testAuc change very little even when the use of the dataset decreases.   

if __name__ == "__main__":
    main()
