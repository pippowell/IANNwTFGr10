import mlp
import matplotlib.pyplot as plt
import dataset

# Create list to store errors for later plotting
losslist = []

# Running through 1000 epochs
while e < 1001:

    for i in dataset.x:
        input = array(dataset.x[i])
        target = dataset.t[i]
        mlp = MLP(1, 10, input, 1, target=target)
        loss = mlp.loss
        losslist.append(loss)

    e = e + 1

# Visualizing Training
plt.title("Visualizing Training")
plt.plot(index, 0-1000, '.', color="red", label='Output')
plt.plot(index, losslist, '.', color="green", label='Target')

plt.xlabel("Epoch")
plt.ylabel("Error/Loss")
plt.legend();

plt.show()
