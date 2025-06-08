from utils.model_builder import model_builder
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping
from utils.show_train_results import show_train_results
from utils.load_preprocess import load_preprocess

# Hypertunnig

([X_train, Y_train], [X_val, Y_val], [X_test, Y_test]) = load_preprocess()

tuner = kt.Hyperband(
    model_builder,
    objective='val_accuracy',
    max_epochs=20,
    factor=2,
    directory="train_results",
    project_name="Fashion",
)

print(" ~~~~~~~~ Espacio de busqueda de hiperparametros")
tuner.search_space_summary()

tuner.search(
    X_train,
    Y_train,
    validation_data=(X_val, Y_val),
)


best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Mejores hiperparametros")
print(best_hps.values)

early_stopping = EarlyStopping(
    monitor='val_accuracy',  # Métrica a monitorear
    mode='max',              # Queremos maximizar el Accuracy
    patience=15,             # Número de épocas sin mejora antes de detener
    verbose=1,
    restore_best_weights=True  # Crucial: recupera los pesos del mejor modelo
)


best_model = tuner.hypermodel.build(best_hps)
hist=best_model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=20,
    callbacks=[early_stopping]
)

test_loss, test_accuracy = best_model.evaluate(X_test, Y_test)
print(f"Loss en test: {test_loss}")
print(f"Accuracy en test: {test_accuracy}")


show_train_results(hist)
best_model.save("model.keras")
