import numpy as np

array = np.random.normal(loc = 0, scale = 1, size = (5,5))
# loc: the mean of the distribution, scale: the standard deviation of the distribution, size: the shape of the numpy array
# 1. Create a 5x5 NumPy array filled with normally distribution (i.e. µ = 0 (mean), σ = 1 (standard deviation)).

print("the array after step 1:")
print(array)

secondarray = array # so that array won't change after multiple code runs

num_rows = len(secondarray)     # length of the row
num_cols = len(secondarray[0])  # length of the column

for row in range(num_rows):
    for col in range(num_cols): 
        if (secondarray[row][col] > 0.09):
            # 2. If the value of an entry is greater than 0.09, replace it with its square. 
            secondarray[row][col] = secondarray[row][col]*secondarray[row][col]
        else: 
            # Else, replace it with 42.
            secondarray[row][col] = 42

print("the array after step 2:")
print(secondarray)

# 3. Use slicing to print just the fourth column of your array.
final_array = secondarray[:, 4:]      # <slice> = <array>[start_row:end_row, start_col:end_col]

print("the array after step 3:")
print(final_array)
