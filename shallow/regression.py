import numpy as np


#? Regression Algorithms
# note: returning [nan] array of weights. :: Solved --> Issue was occuring due to Non Numerical Dataset.
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
