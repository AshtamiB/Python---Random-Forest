# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 06:50:40 2018

@author: Ashtami
"""

import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tools import categorical
from sklearn import tree
# ------------------ Loading Dataset --------------------------#
dataframe = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data")
dataframe = dataframe .drop(dataframe .index[:-1000])
numericdf = dataframe[dataframe .columns[1:9]]
categordf = dataframe[dataframe .columns[0]]
categordf_en = categorical(categordf.values , drop=True)
categordf_en = categordf_en[:, 0:2]
numeric_arr = np.asarray(numericdf.values)
categor_arr = np.asarray(categordf_en)
Output = numeric_arr[:, 7]
Input_numeric = numeric_arr[:, 0:6]
Input_categor = categor_arr
Input = np.concatenate((Input_numeric, Input_categor), axis=1)
#---------------------------------------------------------------#
RF = RandomForestClassifier(n_estimators=5, random_state=12)
RF.fit(Input, Output)
Z_RF = RF.predict(Input)
CM_RF= confusion_matrix(Output, Z_RF)
#---------------------------------------------------------------
DT = tree.DecisionTreeClassifier()
DT.fit(Input, Output)
Z_DT = DT.predict(Input)
CM_DT= confusion_matrix(Output, Z_DT)
#---------------------------------------------------------------
plt.subplot(121)
sns.heatmap(CM_RF.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.subplot(122)
sns.heatmap(CM_DT.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

################################################
#EXAMPLE 2
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import svm, datasets
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
# ------------------ Loading Dataset --------------------------
iris = datasets.load_iris()
X = iris.data[:, :2] # we only take the first two features.
y = iris.target
# ------------- create a mesh to plot in ----------------------
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))
X_for_plot = np.c_[xx.ravel(), yy.ravel()]
# ------------- Create the RF object --------------------------
RF = RandomForestClassifier(n_estimators=3, random_state=12)
RF.fit(X, y)
Z = RF.predict(X_for_plot)
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 5))
plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('Random Forest')
plt.show()