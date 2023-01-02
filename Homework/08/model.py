import tensorflow as tf
import dataset 

class encoder(tf.keras.Model):
    def __init__(self, embedding):
        super(encoder, self).__init__()

        # convolutional layers with stride 1 followed by a pooling layer
        self.convlayer1 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu', stride=1)
        self.maxpool1 = tf. tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='same')
        
        self.convlayer2 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu', stride=1)
        self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='same')
        
        # flatten the feature maps and use a dense layer to produce an embedding of a certain size
        self.flatten =  tf.keras.layers.Flatten()
        self.output = tf.keras.layers.Dense(embedding, activation='relu')

    def __call__(self, input):

        x = self.convlayer1(input)
        x = self.maxpool1(x)
        x = self.convlayer2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.output(x)

        return x

class decoder(tf.keras.Model):
    def __init__(self):
        super(decoder, self).__init__()

        # Use a dense layer to restore the dimensionality of the flattened feature maps from the encode
        self.restore_di_layer = tf.keras.layers.Dense(784, activation='sigmoid')

        # Reshape the resulting vector into feature maps
        self.reshape_layer = tf.keras.layers.Reshape((28, 28))

        # Use upsampling or transposed convolutions to mirror your encoder.
        self.convTlayer1 = tf.keras.layers.Conv2DTranspose(filters=48, kernel_size=3, padding='same', activation='relu', stride=1)
        self.convTlayer2 = tf.keras.layers.Conv2DTranspose(filters=48, kernel_size=3, padding='same', activation='relu', stride=1)
        
        # As an output layer, use a convolutional layer with one filter and sigmoid activation to produce an output image
        self.output = tf.keras.layers.Conv2D(filters=1, activation='sigmoid')

    def __call__(self, input):

            x = self.restore_di_layer(input)
            x = self.reshape_layer(x)
            x = self.convTlayer1(x)
            x = self.convTlayer2(x)
            x = self.output(x)

            return x

class autoencoder(tf.keras.Model): 
    def __init__(self):
        super(autoencoder, self).__init__()

        self.encoder = encoder(embedding=10)
        self.decoder = decoder()

    def __call__(self, input):
        
        encoded = self.encoder(input)
        decoded = self.decoder(encoded) 
        
        return decoded

