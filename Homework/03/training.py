import mnisttf
import network
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 2.4 Training the network

# hyperparameters
epoch = 10
learning_rate = 0.1

# model object
model = network.MyModel()

# choose optimizer and loss
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
loss_func_categorical = tf.keras.losses.CategoricalCrossentropy()

# different arrays for the different values for visualization
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# define training and test datasets
train_dataset = mnisttf.train_ds.apply(mnisttf.prepare_data)
test_dataset = mnisttf.test_ds.apply(mnisttf.prepare_data)

# print("train data set: ", train_dataset)
# >> train data set:  <PrefetchDataset element_spec=(TensorSpec(shape=(None, 784), dtype=tf.float32, name=None), 
# >>                   TensorSpec(shape=(None, 10), dtype=tf.float32, name=None))>"

def train_step(model, input, target, loss_func, optimizer):

    #train_step
    with tf.GradientTape() as tape:
        prediction = model(input)
        loss = loss_func(target, prediction)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss

# test over complete test data
def test(model, test_data, loss_function):

    test_accuracy_aggregator = []
    test_loss_aggregator = []

    for (input, target) in test_data:
        prediction = model(input)
        sample_test_loss = loss_function(target, prediction)
        sample_test_accuracy = np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())
        test_accuracy_aggregator.append(sample_test_accuracy)

    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

    return test_loss, test_accuracy

tf.keras.backend.clear_session()
test_dataset = test_dataset.take(1000)
train_dataset = train_dataset.take(10000)

# testing once before we begin
test_loss, test_accuracy = test(model, test_dataset, loss_func_categorical)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

# check how model performs on train data once before we begin
train_loss, train_accuracy = test(model, train_dataset, loss_func_categorical)
train_losses.append(train_loss)
train_accuracies.append(train_accuracy)

# We train for num_epochs epochs. NOTICE WE DO THIS ONLY ON A TINY FRACTION OF THE DATA!
def train(epoch, model, traindata, testdata, lossfunction, optimizer):

    for e in range(epoch):
        print(f'Epoch: {str(e)} starting with test accuracy {test_accuracies[-1]}')

        #training (and checking in with training)
        #ONLY TAKING A TINY FRACTION OF THE DATA!

        epoch_loss_agg = []
        epoch_ac_agg = []

        for input,target in train_dataset:
            prediction = model(input)
            train_loss = train_step(model, input, target, lossfunction, optimizer)
            train_accuracy = np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
            train_accuracy = np.mean(train_accuracy)
            epoch_loss_agg.append(train_loss)
            epoch_ac_agg.append(train_accuracy)

        #track training loss
        train_losses.append(tf.reduce_mean(epoch_loss_agg))
        train_accuracies.append(tf.reduce_mean(epoch_ac_agg))

        #testing, so we can track accuracy and test loss
        test_loss, test_accuracy = test(model, testdata, lossfunction)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

# Run the network
train(epoch,model,train_dataset,test_dataset,loss_func_categorical,optimizer)

# Visualization
plt.figure()
line1, = plt.plot(train_losses)
line2, = plt.plot(test_losses)
line3, = plt.plot(train_accuracies)
line4, = plt.plot(test_accuracies)
plt.xlabel("Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend((line1,line2,line3,line4),("Training Loss","Test Loss","Training Accuracy","Test Accuracy"))
plt.show()

## update on 17.11: 
## train_step, test methods are copied from lecture jnotebook
## I get an error with the batch size, so we need to give a change there.
## I tried running the code without .take(320).batch(32), but then it would run until the 2nd epoch.
## Also we need to get train_accuracy for data visualization - this part was not mentioned in the lecture