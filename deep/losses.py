"""
Author: Satwik Srivastava
Date: 05-02-2021
Losses for the neural network
"""


import numpy as np
# from functional import one_hot_encode

# === LOSSES === #

class MSE2:
    def __init__(self, outputs, labels, model):
        self.outputs = outputs
        self.labels = labels
        self.model = model
        
    def __call__(self):
        return self.forward()
    
    def __normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        
        return v/norm
        
    def forward(self):
        # add loss value to self.output
        pass
        
    def backward(self):
        col = self.model.layers[::-1]
        self.grads = []
        self.bgrads = []
        
        error = col[0].output - self.labels + 0.00001
        for idx, layer in enumerate(col):
            if idx+1 < len(col):
                delta = error*self.model.ders[layer] + 0.00001
                grad = np.dot(col[idx+1].output.T, delta)
                self.__normalize(grad)
                self.grads.append(grad)
                bgrad = np.sum(error)
                self.bgrads.append(bgrad)
                error = np.dot(error, layer.weights.T)
                
        # revresing the gradient                
        self.grads = self.grads[::-1]
        
        
class MSE1:
    def __init__(self, outputs, labels, model):
        self.outputs = outputs
        self.labels = labels
        self.model = model
        
    def __call__(self):
        return self.forward()
    
    def __normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        
        return v/norm
        
    def forward(self):
        # add loss value to self.output
        pass
        
    def backward(self):
        col = self.model.layers[::-1]
        self.grads = []
        error = col[0].output - self.labels + 0.00001  # figure out what outputs we have to use ?
        
        for idx, layer in enumerate(col):
            if idx+1 < len(col):
                delta = error*self.model.ders[layer] + 0.00001
                grad = np.dot(col[idx+1].output.T, delta)
                self.__normalize(grad)
                self.grads.append(grad)
                error = np.matmul(layer.output, self.grads[-1].T)
        
        # revresing the gradient                
        self.grads = self.grads[::-1]


class CategoricalCrossEntropy:
    def __init__(self, outputs, labels, model):
        self.outputs = outputs
        self.labels = labels
        self.model = model
        
    def forward(self):
        self.output = -np.sum(np.dot(self.labels.T, np.log(self.outputs))) 
        
        
    def backward(self):
        col = self.model.layers[::-1]
        self.grad = []
        error = col[0].output - self.labels + 0.00001  # figure out what outputs we have to use ?
        
        for idx, layer in enumerate(col):
            if idx+1 < len(col):
                delta = error*self.model.ders[layer] + 0.00001
                self.grad.append(np.dot(col[idx+1].output.T, delta))
                error = np.matmul(layer.output, self.grad[-1].T)
        