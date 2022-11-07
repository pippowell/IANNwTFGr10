import numpy as np
import matplotlib.pyplot as plt

# 1. Randomly generate 100 numbers between 0 and 1 and save them to an array ’x’. These are your input values.
# DECISION NEEDED: Float or integer numbers here?
x = np.random.random_sample((100))
print(x)

#Alternatives
# x = np.random.uniform(0,1,size=100)
# x = np.random.sample(100)
# if integer x = np.random.randint(low=0,high=1,size=100)

# # 2. Create an array ’t’. For each entry x[i] in x, calculate x[i]**3-x[i]**2 and save the results to t[i]. These are your targets.
t = []
for element_x in range(0, len(x)):
    element_t = x[element_x]**3 - x[element_x]**2 
    t.append(element_t)
print(t)

#Alternative
#t = (x**3) - (x**2)
#print(t)

# Optional: Plot your data points along with the underlying function which generated them. 
# Wooki: not sure what they mean by "along with the underlying function which generated them"
index = range(0, 100)

plt.title("Optional task")
plt.plot(index, x, '.', color="red", label='Output')
plt.plot(index, t, '.', color="green", label='Target')

plt.xlabel("Index")
plt.ylabel("Target Value")
plt.legend();

plt.show()