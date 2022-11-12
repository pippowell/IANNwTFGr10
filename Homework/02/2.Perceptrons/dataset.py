import numpy as np
import matplotlib.pyplot as plt

# Randomly generate 100 numbers between 0 and 1 and save them to an array ’x’. These are your input values.
x = np.random.random(100)

# Create an array ’t’. For each entry x[i] in x, calculate x[i]**3-x[i]**2 and save the results to t[i]. These are your targets.
# Note - adding one to end of function to account for ReLU not being able to model functions with values less than 0 (negative numbers) as ReLU does not have negative values.
t = (x**3) - (x**2) + 1

# Optional: Plot your data points along with the underlying function which generated them. 
# Wooki: not sure what they mean by "along with the underlying function which generated them"
index = range(0, 100)

plt.title("Visualizing Random Data")
plt.plot(index, x, '.', color="red", label='Output')
plt.plot(index, t, '.', color="green", label='Target')

plt.xlabel("Index")
plt.ylabel("Target Value")
plt.legend();

plt.show()
