import keras
from keras import layers

# Creating our first Neuron
simple_model = keras.Sequential([
    keras.Input(shape=(3,)),
    layers.Dense(units=1)
])

summary = simple_model.weights

# Deeper capacity model
layered_model = keras.Sequential([
    keras.Input(shape=(3,)),
    # The hidden ReLU layers
    layers.Dense(units=4, activation='relu'),
    layers.Dense(units=4, activation='relu'),
    # The linear output layer
    layers.Dense(units=1)
])

# Wider capacity model
wider = keras.Sequential([
    keras.Input(shape=(3,)),
    keras.Dense(12, activation='relu'),
    keras.Dense(1)
])
