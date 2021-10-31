import amphi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X = pd.read_csv("projects\pers_ML_LIB\data\Iris.csv")
# print(X.head())
X = X.drop(["Species", "Id"], axis=1)
# print(X.head())
X_arr = X.to_numpy()
# print('\n',X_arr)
# print(X_arr.shape)
# fig, ax = plt.subplots(figsize=(8,6))
# ax.scatter(x=X_arr[:, 0], y=X_arr[:, 2], cmap="plasma", edgecolors='black')
# plt.show()
# , "SepalWidthCm", "PetalWidthCm"

# Initializing GMM Classifier
pca = amphi.PCA(2)
x_new = pca.reduce(X_arr)
clf = amphi.GMM(2, x_new, 50)
results = clf.start_EM()
clf.plot()

# Initializing Kmeans Classifier
clf2 = amphi.Kmeans()
clf2.predict(x_new)
clf2.plot()
