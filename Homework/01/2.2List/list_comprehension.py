list = [x*x for x in range(100)]
# Using a single line of code, get a list of the squares of each number between 0 and 100

newlist = [x*x for x in range(100) if (x*x) % 2 == 0]
#Then do it again, but only include those squares which are even numbers.

print(list)
print(newlist)
