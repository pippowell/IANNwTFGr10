import tensorflow as tf
from keras import backend as K

# global var
embedding = 10
zmean = tf.keras.layers.Dense(embedding)
zlogvar = tf.keras.layers.Dense(embedding)

class encoder(tf.keras.Model):
    def __init__(self, embedding=embedding, vae=False):
        super(encoder, self).__init__()

        # Alternative: 
        # convolutional layers with stride 1 followed by a pooling layer
        # self.convlayer1 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu', strides=1)
        # self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='same')
        # self.convlayer2 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu', strides=1)
        # self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='same')

        self.vae = vae

        # use convolutional layers with stride 2 for subsampling
        self.convlayer1 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu', strides=2) # shape=(14,14,48)
        self.convlayer2 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu', strides=2) # shape=(7,7,48)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

        # flatten the feature maps and use a dense layer to produce an embedding of a certain size - in our case: 10
        self.flatten =  tf.keras.layers.Flatten() # shape=(2352,1)

        if vae==True:
            self.z_mean = zmean
            self.z_log_var  = zlogvar

            ## option 1
            def sampling(args):
                z_mean, z_log_var = args
                epsilon = K.random_normal(shape=(K.shape(z_mean)[0], embedding),
                                        mean=0., stddev=0.1)
                return z_mean + K.exp(z_log_var) * epsilon

            self.output_layer = tf.keras.layers.Lambda(sampling)

            ### option 2
            # class Sampling(tf.keras.layers.Layer):
            #     """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

            #     def call(self, inputs):
            #         z_mean, z_log_var  = inputs
            #         batch = tf.shape(z_mean)[0]
            #         dim = tf.shape(z_mean)[1]
            #         epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
                    
            #         return z_mean + tf.exp(0.5 * z_log_var) * epsilon

            # self.output_layer = Sampling()([z_mean, z_log_var])

            ### option 3
            # class KLDivergenceLayer(Layer):
            #     """ Identity transform layer that adds KL divergence
            #     to the final model loss.
            #     """

            #     def __init__(self, *args, **kwargs):
            #         self.is_placeholder = True
            #         super(KLDivergenceLayer, self).__init__(*args, **kwargs)

            #     def call(self, inputs):

            #         mu, log_var = inputs

            #         kl_batch = - .5 * K.sum(1 + log_var -
            #                                 K.square(mu) -
            #                                 K.exp(log_var), axis=-1)

            #         self.add_loss(K.mean(kl_batch), inputs=inputs)

            #         return inputs

            # z_mu, z_log_var = KLDivergenceLayer()([z_mean, z_log_var])

            # # normalize log variance to std dev
            # z_sigma = tf.keras.layers.Lambda(lambda t: K.exp(.5*t))(z_log_var)

            # eps = Input(tensor=K.random_normal(shape=(K.shape(x)[0], embedding)))
            # z_eps = Multiply()([z_sigma, eps])

            # self.output_layer = Add()([z_mu, z_eps])


        else: 
            self.output_layer = tf.keras.layers.Dense(embedding, activation='relu') # shape=(10,1)

    @tf.function
    def call(self, input):
        x = self.convlayer1(input)
        x = self.batchnorm1(x)
        x = self.convlayer2(x)
        x = self.flatten(x)

        if self.vae==True:
            z_mean = self.z_mean(x)
            z_log_var = self.z_log_var(x)
            x = self.output_layer(([z_mean, z_log_var]))
        else:
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

    @tf.function
    def call(self, input):

            x = self.restore_di_layer(input) # shape=(bs,784)
            x = self.reshape_layer(x) # shape=(bs,28,28,1)
            x = self.convTlayer1(x) # shape=(bs,28,28,48)
            x = self.convTlayer2(x) # shape=(bs,28,28,48)
            x = self.batchnorm1(x)
            x = self.output_layer(x) # (bs,28,28,1)

            return x

class autoencoder(tf.keras.Model): 
    def __init__(self, vae=False):
        super(autoencoder, self).__init__()

        # Define separate models for the encoder and decoder and initialize them in the autoencoder constructor
        self.encoder = encoder(embedding=10, vae=vae)
        self.decoder = decoder()

    @tf.function
    def call(self, input, training=False):
        
        encoded = self.encoder(input)
        decoded = self.decoder(encoded) 
        
        return decoded
