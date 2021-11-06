"""
Author: Satwik Srivastava
Profiles >>==>
    LinkedIn: BLANK
    github: BLANK

This is a personal project that has some of the most common Machine Learning algorithms implemented into it.
All the algorithms are implemented from scratch using basic libraries such as numpy, pandas and matplotlib.
"""

# todo: Implement identity and diagonal covariance matrices in GMM
# todo: Implement returning labels in GMM
#// todo: Complete plot functions for the models :: completed


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_spd_matrix
from matplotlib.patches import Ellipse
from scipy import stats

#* General Functions


def mse(y_pred, y, give_rmse=False, repr=False):
    mse = np.mean((y_pred - y)**2)
    rmse = np.sqrt(mse)

    if not give_rmse and not repr:
        return mse
    elif not give_rmse and repr:
        return f"The Mean Squared Error is: {mse}"
    elif give_rmse and not repr:
        return mse, rmse

    return f"The Mean Squared Error is: {mse}, The Root Mean Squared Error is: {rmse}"

#? Regression Algorithms
# note: returning [nan] array of weights.
##* Linear Regression


class LinearRegerssion:
    def __init__(self, lr=0.01, n_iters=100, use_sgd=False):
        self.lr = lr  # learning rate used in optimisation using the gradient descent algorithm
        self.n_iters = n_iters
        self.use_sgd = use_sgd
        self.weights = None
        self.bias = 0.
        self.predictions = None

    # fit method to use training data to approximate the weights and biases
    def fit(self, X, y):
        n_samples, n_features = X.shape
        # initialize all the weights as zeroes
        self.weights = np.zeros(n_features, dtype='float64')

        # gradient descent algorithm to update weights
        # can be also taken as an update rule
        for _ in range(self.n_iters):
            # initial predictions
            y_hat = np.dot(X, self.weights) + self.bias
            # calculating gradients
            # differentiation of the sq. error loss wrt weight
            dw = (1/n_samples)*(np.dot(X.T, (y_hat - y)))
            # differentiation of the sq. error loss wrt bias
            db = (1/n_samples)*(np.sum(y_hat - y))
            # updating using \alpha or learning rate
            self.weights -= self.lr*dw
            self.bias -= self.lr*db

    # make predictions using this method
    def predict(self, X):
        self.predictions = np.dot(X, self.weights) + self.bias
        return self.predictions

    # plot your regression model
    # todo
    def plot(self, X, X_train, X_test, y_train, y_test):
        # note: add PCA to reduce given data into 1D
        y_predict_line = np.dot(X, self.weights) + self.bias
        # print(X.shape, y_predict_line.shape)

        cmap = plt.get_cmap('Blues')
        fig = plt.figure(figsize=(15, 10))
        m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=27)
        m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=27)
        plt.plot(X, y_predict_line, color='black',
                 linewidth=2.5, label='Prediction')
        
        plt.title("Linear Regression Plot")
        plt.show()


#? classification Algorithms
##* Logistic Regression
"""
## here the given tranformation wx + b is passed through a squashing function
## which squashes the output i.e. reduces it to a given domain or set.
## here we use the sigmoid function which is 1/(1 + exp(-(wx + b)))
## these squashed values can also be interpreted as a probability of a given feature -> label 
## here our cost function is the cross entropy function
"""


class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000, use_sgd=False):
        self.lr = lr
        self.n_iters = n_iters
        self.use_sgd = use_sgd
        self.weights = None
        self.bias = None
        self.predictions = None
        self.probs = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # initializing weights and biases
        self.weights = np.zeros(n_features, dtype='float64')
        self.bias = 0.

        # gradient descent
        if not self.use_sgd:
            for _ in range(self.n_iters):
                values = np.dot(X, self.weights) + self.bias
                y_hat = self._sigmoid(values)
                temp = y_hat - y

                dw = (1/n_samples)*np.dot(X.T, temp)
                db = (1/n_samples)*np.sum(temp)

                self.weights -= self.lr*dw
                self.bias -= self.lr*db

    def _sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    #// todo: Overflow warning while using the Sigmoid :: Solved Changed dtype
    def predict(self, X):
        values = np.dot(X, self.weights) + self.bias
        self.probs = self._sigmoid(values)
        labels = [1 if i > 0.5 else 0 for i in self.probs]
        labels = np.array(labels)
        self.predictions = labels
        return labels

    def accuracy(self, labels, y_test, log=False):
        count = 0
        size = len(labels)
        for i in range(size):
            if labels[i] == y_test[i]:
                count += 1

        accuracy = count/size*100
        if log:
            print(
                f"\nThe accuracy of our Regressor is: \033[1m\033[92m{accuracy}\033[0m")
        else:
            return accuracy

    #// todo: complete
    def plot(self, test):
        z = test.ravel()
        z = z[:self.predictions.shape[0]]
        plt.figure(1, figsize=(12, 9))
        plt.clf()
        plt.scatter(z, self.predictions, edgecolor="black", alpha=0.7)

        plt.title("Logistic Regression Plot")
        plt.ylabel("y")
        plt.xlabel("X")

        plt.show()


#? Clustering Algorithms
##* Kmeans
def euclidian_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))


class Kmeans:
    def __init__(self, K=3, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        self.EPS = 1e-4

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # mean of feature vectors for each clusters
        # start with random initialization
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # lets initialize our centroids
        # choose random indexes from the data from (0, n_samples in the data)
        random_sample_idxs = np.random.choice(
            self.n_samples, self.K, replace=False)  # vector of size K, default 3*1
        self.centroids = [self.X[idx]
                          for idx in random_sample_idxs]  # K*1 vector

        # optimization
        for _ in range(self.max_iters):
            # update clusters
            self.clusters = self._create_clusters(self.centroids)
            if self.plot_steps:
                self.plot()

            # update centroids
            centroids_old = self.centroids
            ## since cluster is a list of indices of X in the cluster of centroid
            self.centroids = self._get_centroids(self.clusters)

            # check for convergence
            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        # return cluster labels
        return self._get_cluster_labels(self.clusters)

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, num in enumerate(self.X):
            centroid_idx = self._closest_centroid(num, centroids)
            clusters[centroid_idx].append(idx)

        return clusters

    def _closest_centroid(self, num, centroids):
        distances = [euclidian_distance(num, centroid)
                     for centroid in centroids]  # list of size K
        closest_idx = np.argmin(distances)

        return closest_idx  # returns value between 0, K-1

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean

        return centroids

    def _is_converged(self, centroids_old, centroids):
        distances = [euclidian_distance(
            centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) <= self.EPS

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx

        return labels

    def plot(self):
        fig, ax = plt.subplots(figsize=(8, 6))

        for i, idx in enumerate(self.clusters):
            point = self.X[idx].T
            ax.scatter(*point)

        for centroid in self.centroids:
            ax.scatter(*centroid, marker="x", color="black", linewidths=2)

        plt.show()


##* GMM
### initialize 2D mixtures
def init_mixtures(centres, samples=100, r=10, plot=False, return_aux=False):
    # initialize cluster means
    means = []
    for _ in range(centres):
        mean = np.random.choice(r*10, 2)/10
        means.append(list(mean))

    # initialize cluster covariances
    covs = []
    for _ in range(len(means)):
        covs.append(make_spd_matrix(2))

    # generate multivariate normal data
    X = []
    for mean, cov in zip(means, covs):
        x = np.random.multivariate_normal(mean, cov, samples)
        X += list(x)

    # making array
    X = np.array(X)
    np.random.shuffle(X)

    if plot:
        plt.scatter(X[:, 0], X[:, 1], alpha=0.7, edgecolor='black')

    if return_aux:
        return X, means, covs

    return X

##* Gaussian Mixture Model Based EM Algorithm


class GMM:
    def __init__(self, k, X, n_iters=15, prior=None, cm='full'):
        self.k = k
        self.X = X
        self.n_iters = n_iters
        self.means = None
        self.cm = cm
        cms = ['full', 'diag', 'identity']
        if self.cm not in cms:
            raise ValueError("Invalid argument for Covariance Matrix Type!")

        self.covs = None
        self.EPS = 1e-8

        if prior:
            self.prior = prior
        else:
            self.prior = np.ones(self.k)/self.k

    def get_means_covs(self):
        means = np.random.choice(self.X.flatten(), (self.k, self.X.shape[1]))
        covs = []
        for _ in range(self.k):
            covs.append(make_spd_matrix(self.X.shape[1]))
        covs = np.array(covs)

        return means, covs

    def start_EM(self):
        self.means, self.covs = self.get_means_covs()
        for step in range(self.n_iters+1):
            # Expectation Step
            likelihood = []
            for j in range(self.k):
                likelihood.append(stats.multivariate_normal.pdf(
                    x=self.X, mean=self.means[j], cov=self.covs[j]))

            likelihood = np.array(likelihood)
            assert likelihood.shape == (self.k, len(self.X))

            responsibilities = []
            for j in range(self.k):
                responsibilities.append((likelihood[j]*self.prior[j])/(np.sum([likelihood[i]*self.prior[i] for i in
                                                                               range(self.k)], axis=0)+self.EPS))

            # Maximization Step:: Update the mean and the covariance
            self.means[j] = np.sum(responsibilities[j].reshape(len(self.X), 1)*self.X, axis=0)\
                / np.sum(responsibilities[j]+self.EPS)
            self.covs[j] = np.dot((responsibilities[j].reshape(len(self.X), 1)*(self.X-self.means[j])).T,
                                  (self.X-self.means[j]))/(np.sum(responsibilities[j]) + self.EPS)

            # Update Prior
            self.prior[j] = np.mean(responsibilities[j])

        #! identity Matrix not working
        # if self.cm == 'identity':
        #     self.covs = []
        #     for _ in range(self.k):
        #         self.covs.append(np.ones((X.shape[1], X.shape[1])))
        #     self.covs = np.array(self.covs)

        #! Not always working
        # for diagonal covariance matrix
        if self.cm == 'diag':
            d = []
            for i in range(self.k):
                d.append(np.diag(np.diag(self.covs[i])))
            self.covs = np.array(d)

        assert self.covs.shape == (self.k, self.X.shape[1], self.X.shape[1])
        assert self.means.shape == (self.k, self.X.shape[1])

        return self.means, self.covs

    #* Idea: We can apply PCA to it to project the data into 2D and then plot it
    def plot(self):
        colors = ['tab:blue', 'tab:orange', 'tab:green',
                  'magenta', 'yellow', 'red', 'brown', 'grey']
        # create a grid for visualization purpose
        x = np.linspace(np.min(self.X[..., 0])-1,
                        np.max(self.X[..., 0]+1), 100)
        y = np.linspace(np.min(self.X[..., 1])-1, np.max(self.X[..., 1]+1), 80)

        X_, Y_ = np.meshgrid(x, y)

        pos = np.array([X_.flatten(), Y_.flatten()]).T

        fig = plt.figure(figsize=(16, int(12)))
        plt.clf()
        plt.title("EM Result")
        axes = plt.gca()

        likelihood = []
        for j in range(self.k):
            likelihood.append(stats.multivariate_normal.pdf(
                x=pos, mean=self.means[j], cov=self.covs[j]))

        likelihood = np.array(likelihood)
        # print(f"Shape of likelihood is: {likelihood.shape}")
        # print(likelihood[:5])
        predictions = np.argmax(likelihood, axis=0)
        for c in range(self.k):
            pred_ids = np.where(predictions == c)
            plt.scatter(pos[pred_ids[0], 0], pos[pred_ids[0], 1],
                        color=colors[c], alpha=0.2, marker='s', edgecolors='none')

        plt.scatter(self.X[:, 0], self.X[:, 1],
                    edgecolors='grey', facecolors='none')

        for j in range(self.k):
            a = self.covs[j][0][0]
            c = self.covs[j][1][1]
            b = self.covs[j][1][0]
            lambda_1 = (a+c)/2 + np.sqrt(((a-c)/2)**2 + b**2)
            theta = 0
            if abs(b) < 0.0001:
                if a >= c:
                    theta = 0
                else:
                    theta = 90
            else:
                math.atan2(lambda_1-a, b)*180/math.pi

            eigenvalues, eigenvectors = np.linalg.eig(self.covs[j])
            # print(f"EigenValues: {eigenvalues}\nEigenVectors: {eigenvectors}")

            ell = Ellipse((self.means[j][0], self.means[j][1]),
                          eigenvalues[0], eigenvalues[1], theta, edgecolor=colors[j],
                          fc='None', lw=5)
            ell.set_alpha(0.5)

            axes.add_patch(ell)

        plt.show()

#? Principal Component Analysis


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.eig_vect = None
        self.eig_values = None
        self.mean = None
        self.sigma = None

    def reduce(self, X):
        # get mean reduced matrix
        B, self.mean = self._get_mean_red(X)

        # calculate the covariance matrix
        C = np.matmul(B.T, B)/B.shape[0]
        self.eig_values, self.eig_vect = np.linalg.eig(C)

        # checking if our eigenvectors are sorted
        idxs = np.argsort(self.eig_values[::-1])
        s_evts = self.eig_vect[idxs]
        assert s_evts.all() == self.eig_vect.all()

        # get the singular value matrix of B
        self.sigma = np.diag(self.eig_values)

        X_new = np.dot(X, self.eig_vect[:, :self.n_components])
        assert X_new.shape == (X.shape[0], self.n_components)

        return X_new

    def _get_mean_red(self, X):
        X_bar = np.mean(X, axis=0)
        B = X - np.ones(X.shape)*X_bar
        # X = X - X_bar
        return B, X_bar

    def get_distortion(self):
        distortion_matrix = self.eig_values[self.n_components:]
        return np.sum(distortion_matrix)


class HMM:
    def __init__(self) -> None:
        self. x = None
