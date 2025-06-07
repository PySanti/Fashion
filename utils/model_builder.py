from tensorflow import keras
from keras import layers
from keras import optimizers

def model_builder(hp):
    net = keras.Sequential()

    n_hidden_layers         = hp.Int('n_hidden_layers', min_value=1, max_value=9, step=1)
    learning_rate           = hp.Choice('learning_rate', values=[1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])


    # input layer
    net.add(layers.Flatten(input_shape=(28,28)))

    # hidden layers
    # busqueda de numero de capas optima
    for a in range(n_hidden_layers):
        # busqueda de numero de neuronas optimo
        net.add(layers.Dense(units=hp.Int(f'layer_units_{a}', min_value=24, max_value=576, step=24), activation='relu'))
    
    # output layer
    net.add(layers.Dense(10, activation='softmax'))

    net.compile(
        loss="categorical_crossentropy", 
        optimizer=optimizers.Adam(learning_rate=learning_rate), 
        metrics=["accuracy"])

    return net





