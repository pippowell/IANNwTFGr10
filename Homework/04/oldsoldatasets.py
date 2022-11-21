
'''
OLD SOLUTION - DO NOT USE :)
for i in train_ds:

    #initialize tf datasets for the two math problems

    greeqfivetrain = []
    subtractiontrain = []

    #create a random index value to select the second digit for the math problems
    randomind = np.random.randint(0,60001)

    #pull the labels (the numbers themselves) from the ds for easy calculation of the correct math output
    firstnum = train_ds[i].label
    secondnum = train_ds[randomind].label

    #run the math for the first task (a + b >= 5)
    answer = firstnum + secondnum
    if answer >= 5:
        sol = True
    else:
        sol = False

    #add a tuple containing the two images (whose labels we just used) and the answer to the dataset for this problem
    greateqfive = tf.tuple(train_ds[i], train_ds[randomind], sol)
    greeqfivetrain.append(greateqfive)

    #same procedure but for the second math problem (a - b)
    answer = firstnum - secondnum
    subtr = tf.tuple(train_ds[i], train_ds[randomind], answer)
    subtractiontrain.append(subtr)

for i in test_ds:
    #initialize tf datasets for the two math problems
    greeqfivetest = []
    subtractiontest = []

    #create a random index value to select the second digit for the math problems
    randomind = np.random.randint(0,60001)

    #pull the labels (the numbers themselves) from the ds for easy calculation of the correct math output
    firstnum = train_ds[i].label
    secondnum = train_ds[randomind].label

    #run the math for the first task (a + b >= 5)
    answer = firstnum + secondnum
    if answer >= 5:
        sol = True
    else:
        sol = False

    #add a tuple containing the two images (whose labels we just used) and the answer to the dataset for this problem
    greateqfive = tf.tuple(train_ds[i], train_ds[randomind], sol)
    greeqfivetest.append(greateqfive)

    #same procedure but for the second math problem (a - b)
    answer = firstnum - secondnum
    subtr = tf.tuple(train_ds[i], train_ds[randomind], answer)
    subtractiontest.append(subtr)

# Step 3 - Batching & Prefetching
# cache

greeqfivetrain = greeqfivetrain.cache()
greeqfivetest = greeqfivetest.cache()
subtractiontrain = subtractiontrain.cache()
subtractiontest = subtractiontest.cache()

# shuffle
greeqfivetrain = greeqfivetrain.shuffle(1000)
greeqfivetest = greeqfivetest.shuffle(1000)
subtractiontrain = subtractiontrain.shuffle(1000)
subtractiontest = subtractiontest.shuffle(1000)

# batch
greeqfivetrain = greeqfivetrain.batch(2**6)
greeqfivetest = greeqfivetest.batch(2**6)
subtractiontrain = subtractiontrain.batch(2**6)
subtractiontest = subtractiontest.batch(2**6)

#prefetch
greeqfivetrain = greeqfivetrain.prefetch(tf.data.AUTOTUNE)
greeqfivetest = greeqfivetest.prefetch(tf.data.AUTOTUNE)
subtractiontrain = subtractiontrain.prefetch(tf.data.AUTOTUNE)
subtractiontest = subtractiontest.prefetch(tf.data.AUTOTUNE)

# return preprocessed datasets
return greeqfivetrain, greeqfivetest, subtractiontrain, subtractiontest
'''