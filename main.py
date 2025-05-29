from tensorflow import keras
import keras_tuner as kt
from utils.show_image import show_image
import pandas as pd

(X_train, Y_train), (X_test, Y_test) =  keras.datasets.fashion_mnist.load_data()


print(pd.Series(Y_train).value_counts())

