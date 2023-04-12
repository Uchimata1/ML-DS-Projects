import argparse
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

def calc_mistakes(yHat, yTrue):
  """
  Calculate the number of mistakes
  that the algorithm makes based on the prediction.
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
  
  args = parser.parse_args()
  # load the train and test data assumes you'll use numpy
  xTrain = file_to_numpy(args.xTrain)
  yTrain = file_to_numpy(args.yTrain)
  xTest = file_to_numpy(args.xTest)
  yTest = file_to_numpy(args.yTest)

  # 3c) Train and eval Naive Bayes
  clf = MultinomialNB(force_alpha=True)
  clf.fit(xTrain, yTrain.T.flatten())
  yHat = clf.predict(xTest)
  err = calc_mistakes(yHat, yTest)
  print("Number of errors (Naive Bayes): " + str(err)) # binary: 105 errors, count: 499 

  # 3d) Train and eval a Logistic Regression
  clf = LogisticRegression(random_state=0, solver = 'lbfgs', max_iter=10000).fit(xTrain, yTrain.T.flatten())
  yHat = clf.predict(xTest)
  err = calc_mistakes(yHat, yTest)
  print("Number of errors (Logistic Regression): " + str(err)) # binary: 48, count: 47


if __name__ == "__main__":
    main()