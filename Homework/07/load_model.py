import tensorflow as tf
import model
from pathlib import Path

# getting the directory
hw_directory = str(Path(__file__).parents[0])
model_folder = 'my_model07'
dir = hw_directory + '/' + model_folder

# loading the model
model = tf.keras.models.load_model(dir, custom_objects={"ourlstm": model.ourlstm,
                                                        "BasicCNN_LSTM": model.BasicCNN_LSTM})

# model summary 
model.summary()
