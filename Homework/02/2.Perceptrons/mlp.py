import numpy as np

def relu(x):
    x = np.where(x>0, x, 0)
    return x

def relu_derivative(x):
    x = np.where(x > 0, 1, 0)
    return x

print(relu(-1))

# the loss function is MSE (as mentioned in 2.4)
def loss(output, target):
    subtraction = output - target
    raised = subtraction**2
    return 0.5*raised

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
        print('the original shape of the weight matrix is')
        print(np.shape(self.weight_matrix))

        # instantiate empty attributes for layer input, layer preactivation and layer activation
        self.layer_input = None
        self.layer_preactivation = None
        self.layer_activation = None

    # 2. A method called ’forward_step’, which returns each unit’s activation (i.e. output) using ReLu as the activation function.
    def forward_step(self, input):
        
        self.layer_input = input
        print('shape of input')
        print(np.shape(self.layer_input))
        print('shape of weight matrix')
        print(np.shape(self.weight_matrix))
        self.layer_preactivation = np.dot(self.layer_input, self.weight_matrix)
        print('shape of layer preac')
        print(np.shape(self.layer_preactivation))
        self.layer_activation = relu(self.layer_preactivation + self.bias_vector)
        print('shape of layer ac')
        print(np.shape(self.layer_activation))
        return self.layer_activation

    # A method called backward_step, which updates each unit’s parameters (i.e. weights and bias).
    def backward_step(self,d_wrw,d_wrb):

        # ∂L/∂activation must be obtained from layer l+1 (or directly from the loss function derivative if l is the output layer).
        # need to change this! as is only works on final layer! -> solved in backward_step_wooks
        #self.deriv_loss_activ = deriv_loss_activ

        # gradient w.r.t. weight

        #self.nabla_w = np.transpose(self.layer_input)@(np.multiply(relu_derivative(self.layer_preactivation), deriv_loss_activ))

        # gradient w.r.t. bias vector
        #self.nabla_b = np.multiply(relu_derivative(self.layer_preactivation), deriv_loss_activ)

        # It makes sense to store layer activations, pre-activations and layer input in attributes 
        # when doing the forward computation of a layer.

        # gradient w.r.t. input
        #self.nabla_input = np.multiply(relu_derivative(self.layer_preactivation),deriv_loss_activ) @ np.transpose(self.weight_matrix)

        h = 0.01 # learning rate (smaller than 0.05)
        # update parameters: 
        # weight matrix
        testmat = h*d_wrw

        self.weight_matrix = self.weight_matrix - h * d_wrw

        # bias vector
        self.bias_vector = self.bias_vector - h * d_wrb

# create a MLP class which combines instances of your Layer class into class MLP
class MLP(Layer):
    
    # n_hidden_layers: how many hidden layers, size_hl: how many units in hl, 
    # size_output: how many units in output layer, input_size: how many units in input layer
    def __init__(self, n_hidden_layers: int, size_hl: int, size_output: int, input_size: int):

        self.n_hidden_layers = n_hidden_layers
        self.size_hl = size_hl
        self.size_output = size_output
        self.input_size = input_size
        self.output = None
        
        # list of layers in the order of: input_layer, hidden_layer_1, ..., hidden_layer_n, output_layer
        self.layers = [] 

        # needs further work - what exactly is going into the layers matrix here needs attention

        for i in range(self.n_hidden_layers):

            # how about i = 0 (input layer)? if we don't need it then the for-loop can be in range(1, n_hidden_layers + 2)

            # the first hidden layer
            if i == 0:
                layer = Layer(self.input_size, self.size_hl)
                self.layers.append(layer)

            # the rest of the hidden layers
            if i != (n_hidden_layers) and i != 0:
                layer = Layer(self.size_hl, self.size_hl)
                self.layers.append(layer)
            
            # the last layer (= the output layer)
            else:
                layer = Layer(self.size_hl, self.size_output)
                self.layers.append(layer)

    # forward propagation
    # A forward_step method which passes an input through the entire network
    def forward_propagation(self, input, target):

        for i in range(self.n_hidden_layers+1):

            if i == 0:
                input = input
            #needs work - figure out how to call activation of previous layer properly
            else:
                input = self.layers[i-1].layer_activation

            current_layer = self.layers[i]
            current_layer.forward_step(input)

        for i in range(self.n_hidden_layers+1):
            print('size of weight matrix end of forward prop in layer' + str(i))
            print(np.shape(self.layers[i].weight_matrix))
        
        self.output = self.layers[self.n_hidden_layers].layer_activation
        return self.output


    # backward propagation
    # A backpropagation method which updates all the weights and biases in the network given a loss value
    def backward_propagation(self, target):
        
        final_deriv_loss = loss_derivative(self.output,target) #* relu_derivative(self.output)

        steps = []
        
        for i in reversed(range(self.n_hidden_layers+1)):

            if i == self.n_hidden_layers:
                #check this!
                ac_deriv = final_deriv_loss
                preac_deriv = relu_derivative(self.layers[i].layer_preactivation)
                steps.append(ac_deriv*preac_deriv)

                t_input = np.transpose(self.layers[i].layer_input)

                d_wrw = t_input * np.prod(steps)  # * relu_derivative(self.output)
                d_wrb = np.prod(steps)

                current_layer = self.layers[i]
                current_layer.backward_step(d_wrw, d_wrb)

            else:
                ac_deriv = np.transpose(self.layers[i+1].weight_matrix)
                preac_deriv = relu_derivative(self.layers[i].layer_preactivation)
                steps.append(ac_deriv * preac_deriv) #change again!


                
                d_wrw = self.layers[i].layer_input * np.prod(steps) # * relu_derivative(self.output)
                d_wrb = np.prod(steps)

                current_layer = self.layers[i]
                current_layer.backward_step(d_wrw, d_wrb)

            for i in range(self.n_hidden_layers + 1):
                print('size of weight matrix end of back prop in layer' + str(i))
                print(np.shape(self.layers[i].weight_matrix))


