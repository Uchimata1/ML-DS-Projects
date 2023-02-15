from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 2a) import dataset
iris = load_iris(as_frame=True)
iris = iris.frame
iris.head() # test to see if import worked
iris['target'] = pd.Categorical(iris.target)
iris.dtypes # confirm target is converted to a categorical variable

# 2b) boxplot 
fig1, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, figsize=(7, 5))
fig2, (ax3, ax4) = plt.subplots(ncols=2, sharey=False, figsize=(7, 5))
s_length = sns.boxplot(data = iris, x = 'sepal length (cm)', y = 'target', ax = ax1)
s_width = sns.boxplot(data = iris, x = 'sepal width (cm)', y = 'target', ax = ax2)
p_length = sns.boxplot(data = iris, x = 'petal length (cm)', y = 'target', ax = ax3)
p_width = sns.boxplot(data = iris, x = 'petal width (cm)', y = 'target', ax = ax4)
plt.show()


# 2c) scatter plot
fig3, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, figsize=(10, 5))

sns.scatterplot(data = iris, x = 'sepal length (cm)', y = 'sepal width (cm)', hue = 'target', ax = ax1)
sns.scatterplot(data = iris, x = 'petal length (cm)', y = 'petal width (cm)', hue = 'target', ax = ax2)
plt.show()

# 2d) "rules"
# - (0<=petal_length<3) than it's 0, (3<=petal_length<5) than it's 1, (5<=petal_length) than it's 2
# - (0<=petal_width<0.75) than it's 0, (0.75<=petal_width<1.6) than it's 1, (1.6<=petal_width) than it's 2
