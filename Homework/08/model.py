import tensorflow as tf
import dataset 

class encoder(tf.keras.Model):
    def __init__(self):
        super(encoder, self).__init__()

        self.convlayer1 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu', 
                                                batch_input_shape=(dataset.batch_size, dataset.sequence_len, 28, 28, 1))#, input_shape=(28, 28, 1))
        self.maxpool1 = tf. tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='same')
        self.convlayer2 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu', 
                                                batch_input_shape=(dataset.batch_size, dataset.sequence_len, 28, 28, 1))#, input_shape=(28, 28, 1))
        self.maxpool2 = tf. tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='same')
        # use flatten

    def __call__(self, input):

        x = self.convlayer1(input)
        x = self.maxpool1(x)
        x = self.convlayer2(x)
        x = self.maxpool2(x)

        return x

class decoder(tf.keras.Model):
    def __init__(self):
        super(decoder, self).__init__()

        self.convTlayer1 = tf.keras.layers.Conv2DTranspose(filters=48, kernel_size=3, padding='same', activation='relu', 
                                                batch_input_shape=(dataset.batch_size, dataset.sequence_len, 28, 28, 1))#, input_shape=(28, 28, 1))
        self.convTlayer2 = tf.keras.layers.Conv2DTranspose(filters=48, kernel_size=3, padding='same', activation='relu', 
                                                batch_input_shape=(dataset.batch_size, dataset.sequence_len, 28, 28, 1))#, input_shape=(28, 28, 1))

    def __call__(self, input):

            x = self.convTlayer1(input)
            x = self.convTlayer2(x)

            return x

class autoencoder(tf.keras.Model): 
    def __init__(self):
        super(autoencoder, self).__init__()

        self.encoder = encoder()
        self.decoder = decoder()

    def __call__(self, input):

        output = decoder(encoder(input))

        return output

