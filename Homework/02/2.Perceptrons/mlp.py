import dataset
import numpy as np

def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x > 0 else 0

# the loss function is MSE (as mentioned in 2.4)
def loss(output, target):
    return 0.5*(output - target)**2

# loss derivative w.r.t. output(activation) i.e. ∂L/∂activation
def loss_derivative(output, target):
    return output - target

class Layer: 
    
    # an integer argument ’n_units’, indicating the number of units in the layer, 
    # an integer argument ’input_units’, indicating the number of units in the preceding layer
    def __init__(self, input_units: int, n_units: int):

        #  Instantiate a bias vector and a weight matrix of shape (n inputs, n units). 
        #  Use random values for the weights and zeros for the biases.
        self.bias_vector = np.zeros(n_units)
        self.weight_matrix = np.random.random((input_units, n_units))

        # instantiate empty attributes for layer input, layer preactivation and layer activation
        self.layer_input = None
        self.layer_preactivation = None
        self.layer_activation = None

    # 2. A method called ’forward_step’, which returns each unit’s activation (i.e. output) using ReLu as the activation function.
    def forward_step(self):
        self.layer_input = #?
        self.layer_preactivation = np.dot(self.layer_input, self.weight_matrix)
        self.layer_activation = relu(np.dot(self.layer_input, self.weight_matrix) + self.bias_vector)
        return self.layer_activation

    # A method called backward_step, which updates each unit’s parameters (i.e. weights and bias).
    def backward_step(self):

        # ∂L/∂activation must be obtained from layer l+1 (or directly from the loss function derivative if l is the output layer).
        deriv_loss_activ = loss_derivative(self.layer_activation, dataset.t)

        # gradient w.r.t. weight
        deriv_loss_weight = np.transpose(self.layer_input)@(np.multiply(relu_derivative(self.layer_preactivation), deriv_loss_activ))

        # gradient w.r.t. bias vector
        deriv_loss_bias = np.multiply(relu_derivative(self.layer_preactivation), deriv_loss_activ)

        # It makes sense to store layer activations, pre-activations and layer input in attributes 
        # when doing the forward computation of a layer.

        # gradient w.r.t. input
        deriv_loss_input = np.multiply(relu_derivative(self.layer_preactivation),deriv_loss_activ)@np.transpose(self.weight_matrix)

        h = 0.01 # learning rate (smaller than 0.05)
        # update parameters: 
        # weight matrix
        self.weight_matrix = self.weight_matrix - h*deriv_loss_weight

        # bias vector
        self.bias_vector = self.bias_vector - h*deriv_loss_bias

class MLP:

    # combines instances of your Layer class into class MLP
    def __init__(self, n_hidden_layers: int, size_hl: int, input: array, n_output: int):
        super().__init__(n_units, input_units)

    def forward_propogation(self):

        return loss

    def backward_propogation(self, loss):

    # create input layer based on 'input' variable

    # create hidden layers for i in n_hidden_layers

    # create an output layer with size 'n_output'

    # return output




    # A forward_step method which passes an input through the entire network

    
    # A backpropagation method which updates all the weights and biases in the network given a loss value

