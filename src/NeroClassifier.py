import tensorflow as tf
from tensorflow import keras
import netron

# loss fun:
# need constrative learning loss func

# Create a model
in1 = keras.layers.Input(shape=(1,), dtype='float32')
in2 = keras.layers.Input(shape=(10,), dtype='float32')
in3 = keras.layers.Input(shape=(4,), dtype='float32')
merged = keras.layers.Concatenate(axis=1)([in1, in2, in3])
dense1 = keras.layers.Dense(8, input_dim=3, activation=keras.activations.softmax)(merged)
output = keras.layers.Dense(8, input_dim=3, activation=keras.activations.relu)(dense1)
model = keras.models.Model([in1, in2, in3], output)

print(model.to_json())
