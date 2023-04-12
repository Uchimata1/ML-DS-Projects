import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing


def model_assessment(filename):
    """
    Given the entire data, split it into training and test set 
    so you can assess your different models 
    to compare perceptron, logistic regression,
    and naive bayes. 
    """
    fp = open(filename)
    
    # calculate how many emails are in file
    email_count = 0 
    for line in fp: 
        email_count += 1

    # calc number of line for trainSet using 70/30 train-test split
    trainSet = int(email_count*0.7)

    
    with open(filename, 'r') as f:
        lines_train = []
        lines_test = []

        # trainset
        for i in range(trainSet):
            line = f.readline()
            if not line:
                break
            lines_train.append(line)

        # testSet
        for i in range(trainSet, email_count):
            line = f.readline()
            if not line:
                break
            lines_test.append(line)

    # add train-test split observations to new files
    with open('trainSet.data', 'w') as f:
        f.writelines(lines_train)

    with open('testSet.data', 'w') as f:
        f.writelines(lines_test) 

    return None # GRADING Comment: i added the files to my hw4 folder instead of returning them


def build_vocab_map():
    fp = open("trainSet.data")
    vocab_map = {} # stores the word and frequency as key-value pairs
    line_count = 0

    for line in fp: # loop through lines 
        line_count += 1
        for word in line.split(): # loop through words in each line 
            if word in vocab_map: # if word is in vocab map update its frequency
                vocab_map[word] += 1
            else: 
                if (line_count <= 30 and word != "0" and word != "1"): 
                    vocab_map[word] = 1

    # print(len(vocab_map.keys()))
    # print(vocab_map['debt'])
    return vocab_map


def construct_binary(vocab_map):
    """
    Construct email datasets based on
    the binary representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is 1 if the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """
    # ----------------------------------- Train ----------------------------------- #
    emails_train = open("trainSet.data")
    data = []

    # loop each email and add 1 to feature_vec if vocab word in email else add a 0 to feature_vec
    for email in emails_train: 
        feature_vec = []
        for word in vocab_map.keys(): 
            if word in email:
                feature_vec.append(1)
            else:
                feature_vec.append(0)
        
        data.append(feature_vec)

    binary_train = np.array(data)  
    print(binary_train.shape)  

    # ----------------------------------- Test ------------------------------------- #  
    emails_test = open("testSet.data")
    data = []

    # loop each email and add 1 to feature_vec if vocab word in email else add a 0 to feature_vec
    for email in emails_test: 
        feature_vec = []
        for word in vocab_map.keys(): 
            if word in email:
                feature_vec.append(1)
            else:
                feature_vec.append(0)
        
        data.append(feature_vec)

    binary_test = np.array(data)  
    print(binary_test.shape)  
     

    return binary_train, binary_test


def construct_count(vocab_map):
    """
    Construct email datasets based on
    the count representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is the number of times the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """
    # ----------------------------------- Train ----------------------------------- #
    emails_train = open("trainSet.data")
    data = []

    # loop email and keep track of how many times vocab word appears in each email
    for email in emails_train: 
        feature_vec = []
        for word in vocab_map.keys(): 
            if word in email:
                count = email.count(word) # counts how many times word appears in email
                feature_vec.append(count)
            else:
                feature_vec.append(0)
        
        data.append(feature_vec)

    count_train = np.array(data)  
    print(count_train.shape)   

    # ----------------------------------- Test ------------------------------------- #  
    emails_test = open("testSet.data")
    data = []

    # loop email and keep track of how many times vocab word appears in each email
    for email in emails_test: 
        feature_vec = []
        for word in vocab_map.keys(): 
            if word in email:
                count = email.count(word) # counts how many times word appears in email
                feature_vec.append(count)
            else:
                feature_vec.append(0)
        
        data.append(feature_vec)

    count_test = np.array(data)  
    print(count_test.shape)   

    return count_train, count_test


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default="spamAssassin.data",
                        help="filename of the input data")
    args = parser.parse_args()
    model_assessment(args.data)
    vocab_map = build_vocab_map()
    binary_train, binary_test = construct_binary(vocab_map)
    count_train, count_test = construct_count(vocab_map)

    # extract yTrain and yTest
    emails_train = open("trainSet.data")

    yTrain = []
    for email in emails_train: # train
        if '0' in email:
            yTrain.append(0)
        else: 
            yTrain.append(1)

    emails_test = open("testSet.data")

    yTest = []
    for email in emails_test: #test
        if '0' in email:
            yTest.append(0)
        else: 
            yTest.append(1)

    yTrain = np.array(yTrain)
    yTest = np.array(yTest)
    yTrain = pd.DataFrame(yTrain, columns = ["label"])
    yTest = pd.DataFrame(yTest, columns = ["label"])

    yTrain.to_csv("yTrain.csv", index = False)
    yTest.to_csv("yTest.csv", index = False)
    

    # create dataframe and download as csv for the binary dataset
    binary_train = pd.DataFrame(binary_train, columns = list(vocab_map.keys()))
    binary_train.to_csv("xTrain_binary.csv", index = False)

    binary_test = pd.DataFrame(binary_test, columns = list(vocab_map.keys()))
    binary_test.to_csv("xTest_binary.csv", index = False)

    # create dataframe and download as csv for the count dataset
    count_train = pd.DataFrame(count_train, columns = list(vocab_map.keys()))
    count_train.to_csv("xTrain_count.csv", index = False)

    count_test = pd.DataFrame(count_test, columns = list(vocab_map.keys()))
    count_test.to_csv("xTest_count.csv", index = False)



if __name__ == "__main__":
    main()
