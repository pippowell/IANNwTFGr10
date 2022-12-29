import tensorflow as tf
import model

# config = {'class_name':'ourlstm', 'config':5}
#new_model = tf.keras.models.model_from_config(config)
new_model = tf.keras.models.load_model('my_model07', custom_objects={"ourlstm": model.ourlstm,
                                                                     "BasicCNN_LSTM": model.BasicCNN_LSTM},
                                       compile=False)
print(new_model.summary())
print(new_model.get_config())