import mnisttf
import network
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 2.4 Training the network

# Define the hyperparameters
epoch = 10
learning_rate = 0.1

# Initiate the model
model = network.MyModel()

# Define the optimizer and loss function
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
loss_func_categorical = tf.keras.losses.CategoricalCrossentropy()

# Define arrays for saving values for later visualization
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# Initialize the training and test datasets
train_dataset = mnisttf.train_ds.apply(mnisttf.prepare_data)
test_dataset = mnisttf.test_ds.apply(mnisttf.prepare_data)

# 2.4 Train the Network
def train_step(model, input, target, loss_func, optimizer):

    '''
    :param model: the neural network model object
    :param input: the input to the neural network (tensor)
    :param target: the target value for the input (ideal output of the network)
    :param loss_func: the loss function to be used
    :param optimizer: the optimizer to be used
    :return: the loss at the end of the training step
    '''

    # train the network w/ tf functions
    with tf.GradientTape() as tape:
        prediction = model(input)
        loss = loss_func(target, prediction)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss

# test over complete test data
def test(model, test_data, loss_function):

    '''
    :param model: the neural network model object
    :param test_data: the test data to test the model on
    :param loss_function: the loss function to be used
    :return: the loss and accuracy recordings for the network's application across the inputs
    '''

    # initialize lists to hold the loss and accuracy at each application
    test_accuracy_aggregator = []
    test_loss_aggregator = []

    # feed each input to the model and determine the loss and accuracy of the model for each one
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

# clear the keras backend
tf.keras.backend.clear_session()

# take out a small sample of the overall data in each data set
test_dataset = test_dataset.take(1000)
train_dataset = train_dataset.take(10000)

# test the model once before beginning the main application
test_loss, test_accuracy = test(model, test_dataset, loss_func_categorical)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

# check how model performs on train data once before we begin
train_loss, train_accuracy = test(model, train_dataset, loss_func_categorical)
train_losses.append(train_loss)
train_accuracies.append(train_accuracy)

# train over the given number of epochs with the given params
def train(epoch, model, traindata, testdata, lossfunction, optimizer):

    '''
    :param epoch: the number of epochs to run the model for
    :param model: the neural network model object to train
    :param traindata: the training data to be used
    :param testdata: the test data to be used
    :param lossfunction: the loss function to be used
    :param optimizer: the optimizer to be used
    :return: the records of loss and accuracy for training and testing of the model, for visualization
    '''

    # train the model for the defined number of epochs
    for e in range(epoch):
        print(f'Epoch: {str(e)} starting with test accuracy {test_accuracies[-1]}')

        # create lists to aggregate the loss and accuracy on each training step
        epoch_loss_agg = []
        epoch_ac_agg = []

        # feed each input into the model and record its accuracy and loss for training with this input
        for input,target in traindata:
            prediction = model(input)
            train_loss = train_step(model, input, target, lossfunction, optimizer)
            train_accuracy = np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
            train_accuracy = np.mean(train_accuracy)
            epoch_loss_agg.append(train_loss)
            epoch_ac_agg.append(train_accuracy)

        # add the recorded losses and accuracies to the lists defined earlier
        train_losses.append(tf.reduce_mean(epoch_loss_agg))
        train_accuracies.append(tf.reduce_mean(epoch_ac_agg))

        # test the model and record the losses and accuracies in the lists defined earlier
        test_loss, test_accuracy = test(model, testdata, lossfunction)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    # return the lists for visualization
    return train_losses, train_accuracies, test_losses, test_accuracies

# Run and train the network with the defined parameters
train(epoch,model,train_dataset,test_dataset,loss_func_categorical,optimizer)

# 2.5 Visualize results
plt.figure()
line1, = plt.plot(train_losses)
line2, = plt.plot(test_losses)
line3, = plt.plot(train_accuracies)
line4, = plt.plot(test_accuracies)
plt.xlabel("Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend((line1,line2,line3,line4),("Training Loss","Test Loss","Training Accuracy","Test Accuracy"))
plt.show()