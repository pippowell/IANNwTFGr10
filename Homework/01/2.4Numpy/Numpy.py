import numpy as np

#create 5x5 numpy array with normally distributed numbers
array = np.random.normal(loc = 0, scale = 1, size = (5, 5))
print('The original array: ')
print(array)

#if the value of an entry is > 0.09, replace that entry with its square, else with 42
updatedarray = np.where(array>0.09, array*array, 42)
print('The updated array: ')
print(updatedarray)

#print just the fourth column of the array
print('The fourth column (column index 3) of the updated array: ')
print(updatedarray[:, 3])

# Alternate Solution (No need to grade, just want to have it written down)
# print("the array after step 1:")
# print(array)

# secondarray = array # so that array won't change after multiple code runs

# num_rows = len(secondarray)     # length of the row
# num_cols = len(secondarray[0])  # length of the column

# for row in range(num_rows):
#     for col in range(num_cols): 
#         if (secondarray[row][col] > 0.09):
#             # 2. If the value of an entry is greater than 0.09, replace it with its square. 
#             secondarray[row][col] = secondarray[row][col]*secondarray[row][col]
#         else: 
#             # Else, replace it with 42.
#             secondarray[row][col] = 42

# print("the array after step 2:")
# print(secondarray)

# # 3. Use slicing to print just the fourth column of your array.
# final_array = secondarray[:, 3:4]      # <slice> = <array>[start_row:end_row, start_col(incl.):end_col(excl.)]

# print("the array after step 3:")
# print(final_array)


