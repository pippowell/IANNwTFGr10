import mlp
import matplotlib.pyplot as plt
import dataset
import numpy as np

# Create list to store errors for later plotting
losslist = []
avglist = []

net = mlp.MLP(1, 10, 1, 1) # (1 hidden layer, hidden layers of 10 units, output layer of 1 unit, input layer of 1 unit)

# Running through 1000 epochs
e = 1
while e < 1001:

    for i in range(len(dataset.x)):
        input = np.array([[dataset.x[i]]])
        target = np.array([dataset.t[i]])
        net.forward_propagation(input, target)
        loss = mlp.loss(net.output,target)
        losslist.append(loss)
        net.backward_propagation(target)

    avgloss = np.mean(losslist)
    avglist.append(avgloss)

    print('training completed on epoch ' + str(e) + ' with avgloss ' + str(avgloss))

    e = e + 1

# Visualizing Training
index = range(0,e-1)
plt.title("Visualizing Training Performance")
plt.plot(index, avglist, '.', color="green", label='Error/Loss')

plt.xlabel("Epoch")
plt.ylabel("Error/Loss")
plt.legend();

plt.show()
plt.savefig('Bonus Figures/original.png')
