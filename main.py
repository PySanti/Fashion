from tensorflow import keras
import keras_tuner as kt
from utils.show_image import show_image
import pandas as pd
from sklearn.model_selection import train_test_split

(X_train, Y_train), (X_test, Y_test) =  keras.datasets.fashion_mnist.load_data()
X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=.4, random_state=42, stratify=Y_test)

print("Shape del val")
print(X_val.shape)
print(Y_val.shape)
print("Proporcion de targets del val")
print(pd.Series(Y_val).value_counts())
print("\n\n\nShape del test")
print(X_test.shape)
print(Y_test.shape)
print("Proporcion de targets del test")
print(pd.Series(Y_test).value_counts())


