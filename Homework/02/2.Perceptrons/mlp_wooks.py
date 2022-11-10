import numpy as np

def relu(x):
    x = np.where(x>0, x, 0)
    return x

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

        self.input_units = input_units
        self.n_units = n_units

        #  Instantiate a bias vector and a weight matrix of shape (n inputs, n units). 
        #  Use random values for the weights and zeros for the biases.
        self.bias_vector = np.zeros(n_units)
        self.weight_matrix = np.random.random((input_units, n_units))

        # instantiate empty attributes for layer input, layer preactivation and layer activation
        self.layer_input = None
        self.layer_preactivation = None
        self.layer_activation = None

    # 2. A method called ’forward_step’, which returns each unit’s activation (i.e. output) using ReLu as the activation function.
    def forward_step_wooks(self, input):
        
        output = [] # list to store all the activations, layer by layer
        list_of_preact = [] # list to store all the z vectors, layer by layer

        for b, w in zip(self.bias_vector, self.weight_matrix):
            output.append(relu(np.dot(w * input) + b))

        return output

    # A method called backward_step, which updates each unit’s parameters (i.e. weights and bias).
    def backward_step_wooks(self, input, target):

        # predefine the lists for the gradients w.r.t. bias and weight
        nabla_b = [np.zeros(b.shape) for b in self.bias_vector]
        nabla_w = [np.zeros(w.shape) for w in self.weight_matrix]

        # feedforward
        list_of_act = [] # list to store all the activations, layer by layer
        list_of_preact = [] # list to store all the z vectors, layer by layer

        for b, w in zip(self.bias_vector, self.weight_matrix):
            preact = np.dot(w * input) + b 
            list_of_preact.append(preact)

            act = relu(np.dot(w * input) + b)
            list_of_act.append(act)

        # backward pass
        self.nabla_a = loss_derivative(list_of_act[-1], target) # nabla_a must be obtained from layer l+1, so the input is from layer l

        nabla_b[-1] = np.multiply(relu_derivative(list_of_preact[-1]), self.nabla_a)
        nabla_w[-1] = np.transpose(list_of_act[-2]) @ nabla_b[-1] 
        # input = list_of_act[-2] since nabla_w is using nabla_a w.r.t. the last element of list_of_act.

        # nabla_input is missing (I don't get why it's needed) can someone tell me?

        h = 0.01 # learning rate (smaller than 0.05)
        # update parameters: weight matrix, bias vector
        self.weight_matrix = self.weight_matrix - h * nabla_w
        self.bias_vector = self.bias_vector - h * nabla_b

        # return self.weight_matrix, self.bias_vector

# create a MLP class which combines instances of your Layer class into class MLP
class MLP(Layer):

    #### AttributeError: 'MLP' object has no attribute 'bias_vector' ###
    
    # n_hidden_layers: how many hidden layers, size_hl: how many units in hl, 
    # size_output: how many units in output layer, input_size: how many units in input layer
    def __init__(self, input_units: int, n_units: int, n_hidden_layers: int, size_output: int):
        super().__init__(input_units, n_units)

        # self.bias_vector = np.zeros(n_units)
        # self.weight_matrix = np.random.random((input_units, n_units))

        self.n_hidden_layers = n_hidden_layers
        self.size_output = size_output
        
        # list of layers in the order of: input_layer, hidden_layer_1, ..., hidden_layer_n, output_layer
        self.layers = [] 

        for i in range(0, n_hidden_layers + 1):

            # the first hidden layer
            if i == 0: 
                layer = Layer(input_units, n_units)
                self.layers.append(layer)

            # rest of the hidden layers
            if i != (n_hidden_layers) and i != 0:
                layer = Layer(n_units, n_units)
                self.layers.append(layer)
            
            # the last layer (= the output layer)
            else:
                layer = Layer(n_units, size_output)
                self.layers.append(layer)



    # # A forward_step method which passes an input through the entire network
    # def forward_step_mlp_wooks(self, input):
    #     super().forward_step_wooks(input)

    # # A backpropagation method which updates all the weights and biases in the network given a loss value.
    # def backpropagation_wooks(self, input, target):
    #     super().backward_step_wooks(input, target)

net = MLP(1, 1, 10, 1)
# print(net)

# print(help(MLP))