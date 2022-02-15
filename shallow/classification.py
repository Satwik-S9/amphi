import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


class SVM:
    def __init__(self, colors: dict):
        if colors is not None:
            self.colors = colors
        else:
            self.colors = {1: 'r', -1: 'b'}
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        
        opt_dict = {}
        transforms = [[1, 1],
                      [-1, 1],
                      [-1, -1],
                      [-1, 1]]
        
        self.max_feature_value = max(self.X)
        self.min_feature_value = min(self.X)
        
        self.step_sizes = [self.max_feature_value*0.1, 
                          self.max_feature_value*0.01, 
                          self.max_feature_value*0.001]
       
       # extremely expensive
        b_range_multiple = 5
        b_multiple = 5
        
        latest_optimum = self.max_feature_value*10         
        
        for step in self.step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            
            # we can do this because our problem is convex
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple), 
                                   (self.max_feature_value*b_range_multiple), step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                    
    
    def predict(self, features):
        # sign(x.w + b)
        if not isinstance(features,np.ndarray):
            features = np.array(features)
        if isinstance(features, pd.DataFrame):
            features = features.values
            
        self.predictions = np.sign(np.dot(features, self.weights) + self.bias)
        
    
    def plot(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)
    
