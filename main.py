from keras.src.backend.config import max_epochs
from tensorflow import keras
from sklearn.model_selection import train_test_split
from utils.model_builder import model_builder
import keras_tuner as kt

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


# Hypertunnig

tuner = kt.Hyperband(
    model_builder,
    objective='val_precision',
    max_epochs=10,
    directory="train_results",
    project_name="Fashion"
)

tuner.search_space_summary()

tuner.search(
    X_train,
    Y_train,
    validation_data=(X_val, Y_val),
)


best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Mejores hiperparametros")
print(best_hps)

model = tuner.hypermodel.build(best_hps)
# Entrenar el modelo final con los mejores hiperparámetros
model.fit(X_train, Y_train, epochs=15, validation_data=(X_val, Y_val))

