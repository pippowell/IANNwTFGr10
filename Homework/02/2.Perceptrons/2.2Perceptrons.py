import dataset
import numpy as np


def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x > 0 else 0

# the loss function is MSE (as mentioned in 2.4)
def loss(ouput, target): 
    return 0.5*(ouput - target)**2

# loss derivative w.r.t. output(activation) i.e. ∂L/∂activation
def loss_derivative(ouput, target):
    return ouput - target

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
        self.output = relu(np.dot(self.layer_input, self.weight_matrix) + self.bias_vector)
        return self.output 

    # A method called backward_step, which updates each unit’s parameters (i.e. weights and bias).
    def backward_step(self): 

        # pre-activation is the output of the layer’s matrix multiplication
        self.layer_preactivation = self.layer_input*self.weight_matrix

        # activation is the output of the layer’s activation function
        self.layer_activation = self.output

        # ∂L/∂activation must be obtained from layer l+1 (or directly from the loss function derivative if l is the output layer).
        L_derivative_bzg_activation = loss_derivative(self.ouput, dataset.t)

        # gradient w.r.t. weight
        gradient_weight = np.transpose(self.layer_input)*np.multiply(relu_derivative(self.layer_preactivation), L_derivative_bzg_activation)

        # gradient w.r.t. bias vector
        gradient_bias_vector = np.multiply(relu_derivative(self.layer_preactivation), L_derivative_bzg_activation)

        # It makes sense to store layer activations, pre-activations and layer input in attributes 
        # when doing the forward computation of a layer.

        # gradient w.r.t. input
        gradient_input = np.multiply(relu_derivative(self.layer_preactivation), L_derivative_bzg_activation)*np.transpose(self.weight_matrix)

        h = 0.01 # learning rate (smaller than 0.05)
        # update parameters: 
        # weight matrix
        self.weight_matrix = self.weight_matrix - h*gradient_weight

        # bias vector
        self.bias_vector = self.bias_vector - h*gradient_bias_vector

class MLP:

    # combines instances of your Layer class into into class MLP
    def __init__(self, n_units: int, input_units: int):
        super().__init__(n_units, input_units)

    # A forward_step method which passes an input through the entire network

    
    # A backpropagation method which updates all the weights and biases in the network given a loss value

