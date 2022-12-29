import tensorflow as tf
import model


new_model = tf.keras.models.load_model('my_model07', custom_objects={"ourlstm": model.ourlstm,
                                                                     "BasicCNN_LSTM": model.BasicCNN_LSTM},
                                       compile=False)
print(new_model.summary())
