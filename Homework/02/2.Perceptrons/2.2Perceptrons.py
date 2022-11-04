import numpy as np

class Layer: 
    
    # an integer argument ’n_units’, indicating the number of units in the layer, 
    # an integer argument ’input_units’, indicating the number of units in the preceding layer
    def __init__(self, n_units: int, input_units: int):

        #  Instantiate a bias vector and a weight matrix of shape (n inputs, n units). 
        #  Use random values for the weights and zeros for the biases.
        self.bias_vector = np.zeros(input_units)
        self.weight_matrix = np.random.random((input_units, n_units))

        # instantiate empty attributes for layer input, layer preactivation and layer activation
        self.layer_input = None
        self.layer_preactivation = None
        self.layer_activation = None

    # 2. A method called ’forward_step’, which returns each unit’s activation (i.e. output) using ReLu as the activation function.
    def forward_step(self):
        self.output = max(0, np.dot(self.layer_input, self.weight_matrix) + self.bias_vector)
        return self.output 

    # A method called backward_step, which updates each unit’s parameters (i.e. weights and bias).
    def backward_step(self): 
        