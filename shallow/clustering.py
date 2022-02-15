import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
from scipy import stats
from commons import euclidian_distance
from sklearn.datasets import make_spd_matrix

 
##* Kmeans
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
        # if self.cm == 'diag':
        #     d = []
        #     for i in range(self.k):
        #         d.append(np.diag(np.diag(self.covs[i])))
        #     self.covs = np.array(d)

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
