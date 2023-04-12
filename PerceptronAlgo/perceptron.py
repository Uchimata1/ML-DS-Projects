import argparse
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

class Perceptron():
    mEpoch = 1000  # maximum epoch size
    w = None       # weights of the perceptron

    def __init__(self, epoch):
        self.mEpoch = epoch

    def train(self, xFeat, y):
        """
        Train the perceptron using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the epochs and values the number of mistakes
        """
        stats = {}
        self.w = np.zeros(xFeat.shape[1])

        for Epoch in range(self.mEpoch):             
            error_count = 0 
            index = 0

            for x in xFeat: 
                pred = np.dot(self.w, x)
                if ( (pred >= 0) and (y[index] == 0) ): # predict = 1, when yTrue = 0
                    self.w = self.w - x
                    error_count += 1
                if ( (pred < 0) and (y[index] == 1) ): # predict = 0, when yTrue = 1
                    self.w = self.w + x
                    error_count += 1
                
                index += 1
            
            
            if (error_count == 0): break # stop if no mistakes occur
            stats[Epoch] = error_count 


        return stats

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
            Predicted response per sample
        """
        yHat = []
        for x in xFeat:
            pred = np.dot(self.w, x)
            if (pred >= 0): yHat.append(1)
            else: yHat.append(0)

        return yHat


def calc_mistakes(yHat, yTrue):
    """
    Calculate the number of mistakes
    that the algorithm makes based on the prediction.

    Parameters
    ----------
    yHat : 1-d array or list with shape n
        The predicted label.
    yTrue : 1-d array or list with shape n
        The true label.      

    Returns
    -------
    err : int
        The number of mistakes that are made
    """
    err = 0

    for i in range(len(yHat)): 
        if (yHat[i] != yTrue[i]):
            err += 1
        
    return err


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")
    
    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    np.random.seed(args.seed)   
    model = Perceptron(args.epoch)
    trainStats = model.train(xTrain, yTrain)
    print(trainStats)
    yHat = model.predict(xTest)     
    # print out the number of mistakes
    print("Number of mistakes on the test dataset")
    print(calc_mistakes(yHat, yTest))


    # # 2b) k-fold cross validation
    # k = 5 
    # xData = np.vstack((xTrain, xTest))
    # yData = np.vstack((yTrain, yTest))

    # # shuffle dataset
    # indices = np.random.permutation(xData.shape[0])
    # xData_shuff = xData[indices]
    # yData_shuff = yData[indices]

    # fold_size = xData.shape[0] // k 

    # fold_score = []
    # epoch_arr = []
    # for epoch in range(1, 300, 9):
    #     print(epoch)
    #     epoch_scores = []
    #     for i in range(k):
    #         # get the training and test data for this fold
    #         start_idx = i * fold_size
    #         end_idx = (i + 1) * fold_size
    #         X_test = xData_shuff[start_idx:end_idx]
    #         y_test = yData_shuff[start_idx:end_idx]

    #         X_train = np.vstack((xData_shuff[:start_idx], xData_shuff[end_idx:]))
    #         y_train = np.vstack((yData_shuff[:start_idx], yData_shuff[end_idx:]))

    #         # train and eval model
    #         model = Perceptron(epoch)
    #         model.train(X_train, y_train)
    #         yHat = model.predict(X_test)
    #         err = calc_mistakes(yHat, y_test)
            
    #         # add err to epoch_scores
    #         epoch_scores.append(err)
        
    #     fold_score.append(np.mean(epoch_scores))
    #     epoch_arr.append(epoch)
    
    # min_idx = np.argmin(np.array(fold_score))
    # min_epoch = epoch_arr[min_idx]
    # print("Epoch # with smallest error: " + str(min_epoch))

    # # plot epoch vs avg error 
    # fig, ax = plt.subplots(figsize=(12, 6))

    # ax.plot(epoch_arr, fold_score, marker='o')
    # ax.set_xlabel('Epoch')
    # ax.set_ylabel('Avg. Error')
    # ax.set_title('Epoch vs Avg. Error (Count Dataset)')
    # ax.legend()
    # ax.grid()
    # plt.show()


    # use optimal epoch for final model (binary = 40, count = 289)
    model = Perceptron(40)
    model.train(xTrain, yTrain)
    yHat = model.predict(xTest)  

    # print out the number of mistakes
    print("Number of mistakes on the training dataset: " + str(calc_mistakes(yHat, yTrain))) # binary output = 712, count output = 729
    print("Number of mistakes on the test dataset: " + str(calc_mistakes(yHat, yTest))) # binary output = 53, count ouput = 128


    # 2c) output the 15 words with the most positive weights, and the 15 words with the most negative weights.
    sorted_indices = np.argsort(model.w) # get the indices of sorted array in ascending order
    indices_top15 = sorted_indices[-15:]
    indcies_bottom15 = sorted_indices[:15]

    df = pd.read_csv("xTest_binary.csv")
    words = np.array(list(df.columns)) # extract vocab words 

    top15_words = words[indices_top15]
    bottom15_words = words[indcies_bottom15]

    print("Top 15 most positive weight words: " + str(top15_words))
    print("Bottom 15 most negative weight words: " + str(bottom15_words))




if __name__ == "__main__":
    main()