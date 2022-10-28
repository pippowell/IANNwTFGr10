class Cat:
    # define a ”cat” class in ”cat.py”

    def __init__(self, name):
        # A constructor, which lets you name a ”cat”
        self.name = name

    def greeting(self, anotherself): 
        # A method to make your cats greet each other by first introducing themselves and then addressing a different cat by name
        print("Hello I am " + self.name + ". Nice to meet you " + anotherself.name + ".")
