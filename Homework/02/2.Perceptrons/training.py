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
        net = mlp.MLP(1, 10, 1, 1)
        net.forward_propogation(input,target)
        loss = mlp.loss
        losslist.append(loss)
        net.backward_propogation(loss)

    e = e + 1

# Visualizing Training
plt.title("Visualizing Training")
plt.plot(index, 0-1000, '.', color="red", label='Output')
plt.plot(index, losslist, '.', color="green", label='Target')

plt.xlabel("Epoch")
plt.ylabel("Error/Loss")
plt.legend();

plt.show()
