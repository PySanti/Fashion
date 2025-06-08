from tensorflow import keras
from utils.load_preprocess import load_preprocess
from random import randint
from time import sleep
from utils.show_image import show_image
import numpy as np

model = keras.models.load_model("./model.keras")
target_keys = {
0: "T-shirt",
1: "Pantalón",
2: "Jersey",
3: "Vestido",
4: "Abrigo",
5: "Sandalia",
6: "Camisa",
7: "Zapatilla",
8: "Bolso",
9: "Botín"
}


([X_train, Y_train], [X_val, Y_val], [X_test, Y_test]) = load_preprocess()


while 1:
    rand_index = randint(0, len(X_test))
    rand_image = X_test[rand_index]
    rand_target = np.argmax(Y_test[rand_index])
    prediction = np.argmax(model.predict(np.expand_dims(rand_image, axis=0)))
    print(f"Prediccion del modelo : {target_keys[prediction]}")
    show_image(rand_image*255)
    sleep(3)


