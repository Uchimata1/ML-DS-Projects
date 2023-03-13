import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from collections import Counter
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib import cm

class Node:
    '''
    Helper class which implements a single tree node.
    '''
    def __init__(self, feature=None, threshold=None, left=None, right=None, gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value

class DecisionTree(object):
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion

    def __init__(self, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample
        self.root=None

    def gini(self,s):
        """Compute Gini coefficient of array of values"""
        s=np.array(s, dtype=np.int64)
        diffsum = 0
        for i, xi in enumerate(s[:-1], 1):
            diffsum += np.sum(np.abs(xi - s[i:]))
        return -diffsum / (len(s)**2 * np.mean(s)) 

    def entropy(self, s):
        counts = np.bincount(np.array(s, dtype=np.int64))
        percentages = counts / len(s)

        # Caclulate entropy
        entropy = 0
        for pct in percentages:
            if pct > 0:
                entropy += pct * np.log2(pct)
        return -entropy

    def information_gain(self, parent, left_child, right_child):

        num_left = len(left_child) / len(parent)
        num_right = len(right_child) / len(parent)
    
        return self.entropy(parent) - (num_left * self.entropy(left_child) + num_right * self.entropy(right_child))

    def best_split(self, X, y):
        best_split = {}
        best_info_gain = -1
        n_rows, n_cols = X.shape
        

        for feat_index in range(n_cols): # loop through feat indexes
            X_curr = X[:, feat_index]
            for threshold in np.unique(X_curr): # find optimal threshold 
                df = np.concatenate((X, y.reshape(1, -1).T), axis=1)
                df_left = np.array([row for row in df if row[feat_index] <= threshold])
                df_right = np.array([row for row in df if row[feat_index] > threshold])

                if len(df_left) > 0 and len(df_right) > 0:
                    y = df[:, -1]
                    y_left = df_left[:, -1]
                    y_right = df_right[:, -1]
 
                    if self.criterion=='entropy': 
                        gain = self.information_gain(y, y_left, y_right)
                    else:
                        gain = self.gini(y)
                    if gain > best_info_gain:
                        best_split = {'feature_index': feat_index, 'threshold': threshold,
                            'df_left': df_left, 'df_right': df_right, 'gain': gain}
                        best_info_gain = gain
        return best_split

    def build_tree(self, X, y, depth=0):
        n_rows, n_cols = X.shape
        
        if n_rows >= self.minLeafSample and depth <= self.maxDepth:
            best = self.best_split(X, y)

            if best['gain'] > 0: # if split adds more info than continue tree
                left = self.build_tree( X=best['df_left'][:, :-1], y=best['df_left'][:, -1], 
                    depth=depth + 1
                )
                right = self.build_tree(X=best['df_right'][:, :-1], y=best['df_right'][:, -1], 
                    depth=depth + 1
                )
                return Node(feature=best['feature_index'], threshold=best['threshold'], 
                    left=left, right=right, gain=best['gain']
                )
        return Node(value=Counter(y).most_common(1)[0][0])

    def _predict(self, x, tree):
        if tree.value != None:
            return tree.value

        feature_value = x[tree.feature]
 
        if feature_value <= tree.threshold:
            return self._predict(x=x, tree=tree.left)
        
        if feature_value > tree.threshold:
            return self._predict(x=x, tree=tree.right)


    def train(self, xFeat, y):
        """
        Train the decision tree model.

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
        if type(xFeat).__module__ != np.__name__:
            xFeat = xFeat.to_numpy()
        if type(y).__module__ != np.__name__:
            y = y.to_numpy()

        self.root = self.build_tree(xFeat, y)
        return self


    def predict(self, xFeat):
        if type(xFeat).__module__ != np.__name__:
            xFeat = xFeat.to_numpy()

        yHat = [self._predict(x, self.root) for x in xFeat] # variable to store the estimated class label
        return yHat


def dt_train_test(dt, xTrain, yTrain, xTest, yTest):
    """
    Given a decision tree model, train the model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    dt : DecisionTree
        The decision tree with the model parameters
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
    # train the model
    dt.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = dt.predict(xTrain)
    trainAcc = accuracy_score(yTrain['label'], yHatTrain)
    # predict the test dataset
    yHatTest = dt.predict(xTest)
    testAcc = accuracy_score(yTest['label'], yHatTest)
    return trainAcc, testAcc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("md",
                        type=int,
                        help="maximum depth")
    parser.add_argument("mls",
                        type=int,
                        help="minimum leaf samples")
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

    dt1 = DecisionTree('gini', args.md, args.mls)
    trainAcc, testAcc = dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
    print("Entropy Criterion ---------------")
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)

    ########## 
    dt = DecisionTree('entropy', args.md, args.mls)
    trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
    print("Entropy Criterion ---------------")
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)


    ## Problem 1c: 3D plot of accuracy with respect to depth and minleaf
    entropy_train_acc = []
    entropy_test_acc = []
    depthData = []
    minleafData = []
    for depth in range(2,30,5):
        for minleaf in range(2,30, 5):
            try:
                print("depth, minleaf:", depth, minleaf, "-----------------")          
                dt = DecisionTree('entropy', depth, minleaf)
                trainAcc1, testAcc1 = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
                entropy_train_acc.append(trainAcc1)
                entropy_test_acc.append(testAcc1)
                depthData.append(depth)
                minleafData.append(minleaf)
                print("Accuracy:", trainAcc1, testAcc1)
            except Exception as error:
                print(error)
                continue

    
    # Part 1: entrophy_train_acc 
    ax1 = plt.axes(projection='3d')
    ax1.scatter3D(depthData, minleafData, entropy_train_acc, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax1.set_xlabel('Depth')
    ax1.set_ylabel('MinLeaf')
    ax1.set_zlabel('TrainAcc')
    plt.savefig("trainentropy.png")
    plt.show()
    
    # Part 2: entrophy_test_acc
    ax2 = plt.axes(projection='3d')
    ax2.scatter3D(depthData, minleafData, entropy_test_acc, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax2.set_xlabel('Depth')
    ax2.set_ylabel('MinLeaf')
    ax2.set_zlabel('TestAcc')
    plt.savefig("testentropy.png")
    plt.show()

    # Problem 1d: Time Complexity
        # Train: O(n * d * p^2)
        # bestSplit has a nested for loop that loops through features and worst case all rows. Thus bestplit is O(d * n)
        # buildTree recursively calls itself two times resulting in O(p^2) time complexity  
        # 
        # Predict: O(n*p) 
        # Worst case, predictOne is O(p) because the tree is unlikely to be balanced. Moreover, 
        # predict does a for-loop of n times. Thus our total time complexity for predict is O(n*p) 
    
    
if __name__ == "__main__":
    main()
