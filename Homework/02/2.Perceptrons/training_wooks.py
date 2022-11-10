import mlp
import matplotlib.pyplot as plt
import dataset
import numpy as np

# Create list to store errors for later plotting
losslist = []

# Running through 1000 epochs
e = 1
while e < 1001:

    for i in range(len(dataset.x)):
        input = np.asarray(dataset.x[i])
        target = dataset.t[i]
        net = mlp.MLP(1, 1, 10, 1) # (input layer of 1 unit, 10 hidden layers, hidden layers of 10 units, output layer of 1 unit, input layer of 1 unit)
        net.forward_step_mlp_wooks(dataset.x[i])
        net.backpropagation_wooks(dataset.x[i], dataset.t[i])

    e = e + 1

# Visualizing Training
index = range(0, 100)
plt.title("Visualizing Training")
plt.plot(index, 0-1000, '.', color="red", label='Output')
plt.plot(index, losslist, '.', color="green", label='Target')

plt.xlabel("Epoch")
plt.ylabel("Error/Loss")
plt.legend();

plt.show()
