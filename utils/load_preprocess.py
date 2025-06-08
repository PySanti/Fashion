from tensorflow import keras
from sklearn.model_selection import train_test_split


def load_preprocess():
    # carga de datos
    (X_train, Y_train), (X_test, Y_test) =  keras.datasets.fashion_mnist.load_data()

    # division de test en test y validacion
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=.4, random_state=42, stratify=Y_test)

    # conversion de targets
    Y_train = keras.utils.to_categorical(Y_train, 10)
    Y_test = keras.utils.to_categorical(Y_test, 10)
    Y_val = keras.utils.to_categorical(Y_val, 10)

    # normalizacion
    X_train = X_train / 255.0
    X_val   = X_val / 255.0
    X_test  = X_test / 255.0

    return ([X_train, Y_train], [X_val, Y_val], [X_test, Y_test])



