import numpy as np
from layers import Dense, Flatten
from settings import*


class Sequential:
    def __init__(self, *layers):
        self.layers = list(layers)
        
    def __repr__(self):
        self.total_params = 0
        l1 = 0
        for layer in self.layers:
            l2 = len(repr(layer))
            l1 = max(l1, l2)
        
        
        s = f"{DIVIDER}\n{VERTICAL_S}  Model Summary:\t\t\t\t\t\t\t\t\t\t{VERTICAL_E}\n{DIVIDER}"
        
        for idx, layer in enumerate(self.layers):
            dyn_spc = "".join([" " for _ in range(l1-len(repr(layer))+4)])
            s += f"\n{VERTICAL_S}  {BLUE}Layer {idx:<4}:{END}\t{layer}{dyn_spc}\t{VERTICAL_E}"
            self.total_params += layer.total_params 
            
        s += f"\n{DIVIDER}\n{VERTICAL_S}  total parameters: {self.total_params}\t\t\t\t\t\t\t\t\t{VERTICAL_E}\n{DIVIDER}" 
        
        return s

    def __str__(self):
        return repr(self)
    
    def __call__(self, inputs):
        self.forward(inputs)
        
        return self.output
    
    def __len__(self):
        return len(self.layers)
    
    def summary(self):
        print(repr(self))
    
    def forward(self, inputs: np.ndarray):
        self.ders = {}
        for layer in self.layers:
            layer.forward(inputs)
            inputs = layer.output
            self.output = layer.output
            self.ders[layer] = self.__derivative(layer.activation)
            
    def __derivative(self, activation):
        if activation == 'relu':
            return np.heaviside(self.output, 1)
        
        elif activation == 'sigmoid':
            return self.output*(1-self.output)
        
        elif activation == 'softmax':
            s1 = np.diag(self.output)
            s2 = self.output*self.output.T
            
            return s1 + s2
        
        elif activation == 'tanh':
            return 1 - np.square(self.output)
        
        else:
            return 1