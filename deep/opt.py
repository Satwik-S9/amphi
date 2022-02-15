import numpy as np
from models import Sequential

# === OPTIMIZERS === #
class SGD:
    def __init__(self, model: Sequential, lossfn, lr=0.001, n_iters=1000):
        self.lr = lr
        self.model = model
        self.lossfn = lossfn
    
    def step(self):
        assert len(self.model.layers[1:]) == len(self.lossfn.grads)
        
        for idx, layer in enumerate(self.model.layers[1:]):
            layer.weights -= self.lr*self.lossfn.grads[idx]
            self.model.layers[idx+1] = layer
            
        return self.model.layers
        
        
class Adam:
    def __init__(self):
        pass