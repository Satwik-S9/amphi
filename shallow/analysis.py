import numpy as np


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

