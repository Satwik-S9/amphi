"""
Author: Satwik Srivastava
Comment out whatever block of code you want to test
"""

from math import log
import amphi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, make_regression
from sklearn.model_selection import train_test_split

# X = pd.read_csv("projects\pers_ML_LIB\data\Iris.csv")
# # print(X.head())
# X = X.drop(["Species", "Id"], axis=1)
# # print(X.head())
# X_arr = X.to_numpy()
# # print('\n',X_arr)
# # print(X_arr.shape)
# # fig, ax = plt.subplots(figsize=(8,6))
# # ax.scatter(x=X_arr[:, 0], y=X_arr[:, 2], cmap="plasma", edgecolors='black')
# # plt.show()
# # , "SepalWidthCm", "PetalWidthCm"

# # Initializing GMM Classifier
# pca = amphi.PCA(2)
# x_new = pca.reduce(X_arr)
# clf = amphi.GMM(2, x_new, 50)
# results = clf.start_EM()
# clf.plot()

# # Initializing Kmeans Classifier
# clf2 = amphi.Kmeans()
# clf2.predict(x_new)
# clf2.plot()

# # Initializing Logistic Regression Classifier
# clf_logr = amphi.LogisticRegression()
# data = load_breast_cancer()
# X, y = data.data, data.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ## training the model
# clf_logr.fit(X_train, y_train)
# labels = clf_logr.predict(X_test)
# print(labels)
# print(amphi.mse(labels, y_test, True, True))
# clf_logr.accuracy(labels, y_test, log=True)


## Initializing Linear Regression
X, y = make_regression(n_samples=1000, n_features=4, bias=2.3, random_state=42)
plt.scatter(X[:, 1], y)
clf_lin = amphi.LinearRegerssion()
