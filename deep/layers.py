import numpy as np
import functional

# Setting seed for consistancy
np.random.seed(42)


#* === LAYERS === *#
class Flatten:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.total_params = 0
        self.activation = 'nan'
        
    def __repr__(self):
        return f"(Flatten Layer,\
  shape: ({()}, {self.input_shape}),  total params: {self.total_params})"
    
    def __str__(self):
        return repr(self)
      
    def forward(self, inputs):
        if inputs.shape == self.input_shape:
            self.output = inputs.ravel()
        
        else:
            self.output = inputs.reshape(inputs.shape[0], -1)
    
    
class Dense:
    def __init__(self, input_shape, output_shape,
                 bias=True, w_init='nan', activation='nan'):
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.bias = bias
        self.w_init = w_init
        self.activation = activation
        
        # check arguments
        if w_init not in ['xavier', 'nan', 'he']:
            raise ValueError('w_init must be one of the following {}'\
                .format(['xavier', 'nan', 'he']))
        
        if activation not in ['softmax', 'relu', 'tanh', 'sigmoid', 'nan']:
            raise ValueError('activation must be one of the following {}'\
                .format(['softmax', 'relu', 'tanh', 'sigmoid', 'nan']))
        
        self.__init_weights()
        
        if bias:
            self.total_params = self.weights.size+self.biases.size
        else:
            self.total_params = self.weights.size
        
    def __repr__(self):
        return f"(Dense Layer, shape: ({self.input_shape}, {self.output_shape}),  \
total params: {self.total_params}, activation: {self.activation})"
  
    def __str__(self):
        return repr(self)
        
    
    def __normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v

        return v / norm
    
    def __init_weights(self):
        if self.w_init == 'xavier':
            self.weights = self.__normalize(np.random.randn(self.input_shape, self.output_shape)*np.sqrt(1/self.input_shape))
            
            if self.bias:
                self.biases = np.zeros((1, self.output_shape))
                
        elif self.w_init == 'he':
            self.weights = self.__normalize(np.random.randn(self.input_shape, self.output_shape)*np.sqrt(2/self.input_shape))
            
            if self.bias:
                self.biases = np.zeros((1, self.output_shape))
        
        else:
            self.weights = self.__normalize(np.random.randn(self.input_shape, self.output_shape))
            
            if self.bias:
                self.biases = np.zeros((1, self.output_shape))
            
    
    def forward(self, inputs):
        if self.bias:
            self.output = np.dot(inputs, self.weights) + self.biases
        else:
            self.output = np.dot(inputs, self.weights)
        
        # when activation is not "nan"
        if self.activation == 'softmax':
            self.output = functional.softmax(self.output)
        
        elif self.activation == 'sigmoid':
            self.output = functional.sigmoid(self.output)
        
        elif self.activation == 'tanh':
            self.output = functional.tanh(self.output)
        
        elif self.activation == 'relu':
            self.output = functional.relu(self.output)
        
        
#* === ACTIVATION LAYERS === *#
class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
    

class Softmax:
    def forward(self, inputs):
        exp_val = np.exp(inputs - np.max(inputs))
        self.output = exp_val/exp_val.sum(axis=0)
        

class Sigmoid:
    def forward(self, inputs):
        self.output = 1/(1+np.exp(-inputs))
        

class Tanh:
    def forward(self, inputs):
        self.output = np.tanh(inputs)