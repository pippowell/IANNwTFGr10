import tensorflow as tf
from keras import backend as K

class encoder(tf.keras.Model):
    def __init__(self, embedding, vae=False):
        super(encoder, self).__init__()

        # convolutional layers with stride 1 followed by a pooling layer
        # self.convlayer1 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu', strides=1)
        # self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='same')
        # self.convlayer2 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu', strides=1)
        # self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='same')

        # use convolutional layers with stride 2 for subsampling
        self.convlayer1 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu', strides=2) # shape=(14,14,48)
        self.convlayer2 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu', strides=2) # shape=(7,7,48)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

        # flatten the feature maps and use a dense layer to produce an embedding of a certain size - in our case: 10
        self.flatten =  tf.keras.layers.Flatten() # shape=(2352,1)


        if vae==True:
            z_mean = tf.keras.layers.Dense(embedding)
            z_log_sigma = tf.keras.layers.Dense(embedding)

            def sampling(args):
                z_mean, z_log_sigma = args
                epsilon = K.random_normal(shape=(K.shape(z_mean)[0], embedding),
                                        mean=0., stddev=0.1)
                return z_mean + K.exp(z_log_sigma) * epsilon

            self.output_layer = tf.keras.layers.Lambda(sampling)([z_mean, z_log_sigma])

        else: 
            self.output_layer = tf.keras.layers.Dense(embedding, activation='relu') # shape=(10,1)
            

        # if we are using a variational autoencoder

    def __call__(self, input):

        x = self.convlayer1(input)
        x = self.batchnorm1(x)
        # x = self.maxpool1(x)
        x = self.convlayer2(x)
        # x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.output_layer(x)

        return x

class decoder(tf.keras.Model):
    def __init__(self):
        super(decoder, self).__init__()

        # Use a dense layer to restore the dimensionality of the flattened feature maps from the encoder
        self.restore_di_layer = tf.keras.layers.Dense(784, activation='sigmoid') # shape=(784,1)

        # Reshape the resulting vector into feature maps
        self.reshape_layer = tf.keras.layers.Reshape((28, 28, 1)) 
        
        # Use upsampling or transposed convolutions to mirror your encoder.
        self.convTlayer1 = tf.keras.layers.Conv2DTranspose(filters=48, kernel_size=3, padding='same', activation='relu', strides=1) 
        
        self.convTlayer2 = tf.keras.layers.Conv2DTranspose(filters=48, kernel_size=3, padding='same', activation='relu', strides=1) 
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        
        # As an output layer, use a convolutional layer with one filter and sigmoid activation to produce an output image
        self.output_layer = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='same', strides=1, activation='sigmoid')

    def __call__(self, input):

            x = self.restore_di_layer(input) # shape=(bs,784)
            x = self.reshape_layer(x) # shape=(bs,28,28,1)
            # print(f"done until reshape_layer: {x}")

            x = self.convTlayer1(x) # shape=(ns,28,28,48)
            # print(f"done until convTlayer1: {x}")

            x = self.convTlayer2(x) # shape=(ns,28,28,48)
            # print(f"done until convTlayer2: {x}")

            x = self.batchnorm1(x)

            x = self.output_layer(x) # (bs,28,28,1)
            # print(f"done until outputlayer: {x}")

            return x

class autoencoder(tf.keras.Model): 
    def __init__(self, vae=False):
        super(autoencoder, self).__init__()

        # Define separate models for the encoder and decoder and initialize them in the autoencoder constructor
        self.encoder = encoder(embedding=10, vae=vae)
        self.decoder = decoder()

    def __call__(self, input, training=False):
        
        encoded = self.encoder(input)
        decoded = self.decoder(encoded) 
        
        return decoded

testmodel = autoencoder()
