from tensorflow import keras
from keras import layers
from keras import optimizers
from keras.regularizers import l2

def model_builder(hp):
    net = keras.Sequential()

    n_hidden_layers         = hp.Int('n_hidden_layers', min_value=1, max_value=4, step=1)
    learning_rate           = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5, 1e-6])


    # input layer
    net.add(layers.Flatten(input_shape=(28,28)))

    # hidden layers
    # busqueda de numero de capas optima
    for a in range(n_hidden_layers):
        # busqueda de numero de neuronas optimo
        units_count = hp.Int(f'layer_units_{a}', min_value=24, max_value=624, step=24)
        # busqueda de hiperparametro de regularizacion optimo
        regu = hp.Choice(f'layer_regu_{a}', values=[1e-3, 1e-4, 1e-5, 1e-6])
        # busqueda de dropout rate optimo
        drop = hp.Float(f'layer_drop_{a}', min_value=0.1, max_value=0.3, sampling="log")

        net.add(layers.Dense(units=units_count, activation='relu', kernel_regularizer=l2(regu)))
        net.add(layers.Dropout(rate=drop))
    
    # output layer
    net.add(layers.Dense(10, activation='softmax'))

    net.compile(
        loss="categorical_crossentropy", 
        optimizer=optimizers.Adam(learning_rate=learning_rate), 
        metrics=["accuracy"])

    return net





