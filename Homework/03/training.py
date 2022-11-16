import mnisttf
import network
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense
import tensorflow as tf

# 2.4 Training the network

def train_step(model, input, target, loss_func, optimizer):

    with tf.GradientTape() as tape:
        prediction = model(input)
        loss = loss_func(target, prediction)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

# hyperparameters
epoch = 3 # need to be 10, just set at 3 for testing out the code
learning_rate = 0.1

# model object
model = network.MyModel()

# choose optimizer and loss
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
loss_func = tf.keras.losses.CategoricalCrossentropy()

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

for e in range(0, epoch):
    print(f'Epoch: {str(e)} starting with the train loss of {train_loss[-1]}')

    list = []
    for input, target in train_dataset:
        train_loss = train_step(model, input, target, loss_func, optimizer)
        list.append(train_loss)

    train_losses.append(tf.reduce_mean(list))

# print(train_losses)
# >> [<tf.Tensor: shape=(), dtype=float32, numpy=14.540907>, <tf.Tensor: shape=(), dtype=float32, numpy=14.519989>, 
# >> <tf.Tensor: shape=(), dtype=float32, numpy=14.519989>, <tf.Tensor: shape=(), dtype=float32, numpy=14.51999>, 
# >> <tf.Tensor: shape=(), dtype=float32, numpy=14.519987>, <tf.Tensor: shape=(), dtype=float32, numpy=14.519989>, 
# >> <tf.Tensor: shape=(), dtype=float32, numpy=14.519989>, <tf.Tensor: shape=(), dtype=float32, numpy=14.519991>, 
# >> <tf.Tensor: shape=(), dtype=float32, numpy=14.51999>, <tf.Tensor: shape=(), dtype=float32, numpy=14.519987>]