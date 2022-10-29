from sympy import symbols, diff, exp
# sympy is used to calculate the partial derivative in a mathematical function.

# define sigmoid function
def sigmoid(x):
    return 1 / (1 + exp(-x))

# symbol() function's argument is a string containing symbol which can be assigned to a variable
a, b, x, z = symbols("a, b, x, z", real = True) 
f = 4*a*x**2 + a + 3 + sigmoid(z) + (sigmoid(b))**2

derivx = diff(f, x)
derivz = diff(f, z)
deriva = diff(f, a)
derivb = diff(f, b)

nablaf = derivx + derivz + deriva + derivb

print(diff(f, x))
print(diff(f, z))
print(diff(f, a))
print(diff(f, b))
print(nablaf)

