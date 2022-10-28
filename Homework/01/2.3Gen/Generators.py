def infinite_meow():
    # implement a generator function which returns a meow the first time you call it, 
    x = "Meow "
    while True:
        yield x
        x = x*2
        # and then twice the number of meows on each consecutive call

gen = infinite_meow()
print(next(gen))
print(next(gen))
print(next(gen))
print(next(gen))