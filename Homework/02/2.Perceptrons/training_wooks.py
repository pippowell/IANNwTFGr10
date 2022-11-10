import mlp_wooks
import matplotlib.pyplot as plt
import dataset
import numpy as np

# Create list to store errors for later plotting
losslist = []

# the loss function is MSE (as mentioned in 2.4)
def loss(output, target):
    return 0.5*(output - target)**2

net = mlp_wooks.MLP(1, 1, 10, 1)       # (input layer of 1 unit, 10 hidden layers, hidden layers of 10 units, output layer of 1 unit)

# Running through 1000 epochs
e = 1
while e < 1001:

    for i in range(len(dataset.x)):

        input = dataset.x[i]
        target = dataset.t[i]        
        net.forward_step_wooks(input)
        net.backward_step_wooks(input, target)
        loss = loss(input, target)
        losslist.append(loss)

    e = e + 1

# Visualizing Training
index = range(0, 100)
plt.title("Visualization of the training")
# plt.plot(index, 0-1000, '.', color="red", label='Output')
plt.plot(index, losslist, '.', color="green", label='Target')

plt.xlabel("Epoch")
plt.ylabel("Error/Loss")
plt.legend();

plt.show()
