# Fashion


El objetivo de este proyecto será utilizar el dataset incluido en `keras` llamado `fashion_mnist` y crear una red neuronal capaz de identificar las diferentes prendas de ropa incluidas en este dataset.

Además, implementar técnicas de `hypertunning` a través de la librería `keras tuner`.

Tendré que utilizar una PC de mi clase de programación web para poder realizar este proyecto :|



## Preprocesamiento

### Visualizacion del conjunto de datos

En las primeras observaciones del conjunto, encontramos lo siguiente:

```
Shape del conjunto de entrenamiento
(60000, 28, 28)
(60000,)
Shape del conjunto de entrenamiento
(10000, 28, 28)
(10000,)
```

El conjunto de entrenamiento cuenta con 60.000 registros de matrices de 28x28.\
El conjunto de test cuenta con 10.000 registros de matrices de 28x28.


Las imagenes cuentan con el siguiente formato:


```
[  0   0   0   0   0   0   0   0   0   1   0   0  18 107 119 103   9   0
    0   0   0   0   0   0   0   0   0   0]
```

28 filas, donde cada fila cuenta con un valor entero entre 0 y 255 que representa la escala de gris.

Sabemos que cada registro de matrix de 28x28 representa una prenda de moda. Para revisarlo, creamos una funcion para graficar usando `matplotlib` y encontramos lo siguiente:

![Imagen no encontrada](./images/imagen_1.png)

Los targets son enteros entre 0 y 9:

```
9    6000
0    6000
3    6000
2    6000
7    6000
5    6000
1    6000
6    6000
4    6000
8    6000

```

Cada target se corresponde con lo siguiente:

```
0: T-shirt/top (Camiseta/parte superior)
1: Trouser (Pantalón)
2: Pullover (Jersey)
3: Dress (Vestido)
4: Coat (Abrigo)
5: Sandal (Sandalia)
6: Shirt (Camisa)
7: Sneaker (Zapatilla deportiva)
8: Bag (Bolso)
9: Ankle boot (Botín)
```

### Division del conjunto de datos

Dividimos el conjunto de test en dos para poder contar tambien con un conjunto de validacion. Esto lo lograremos usando la funcion `train_test_split` de `scikit-learn`.

```
Shape del val
(6000, 28, 28)
(6000,)
Proporcion de targets del val
5    600
4    600
6    600
0    600
1    600
8    600
3    600
2    600
9    600
7    600
Name: count, dtype: int64



Shape del test
(4000, 28, 28)
(4000,)
Proporcion de targets del test
0    400
7    400
5    400
3    400
4    400
1    400
8    400
2    400
6    400
9    400

```

### Conversion de targets

Teniendo en cuenta que es un problema de clasificacion multinomial, las neuronas de la output layer deberan implementar `softmax` como funcion de activacion, para ello, debemos modificar el formato de los targets del conjunto, usando la funcion  `keras.utils.to_categorical()`:

```
9 : [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
0 : [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
0 : [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
3 : [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
0 : [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
2 : [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
7 : [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
2 : [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
5 : [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
5 : [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
...
```

A traves del siguiente codigo se transformaron los targets para los 3 conjuntos (validacion, pruebas y entrenamiento):

```
from tensorflow import keras
import keras_tuner as kt
from utils.show_image import show_image
import pandas as pd
from sklearn.model_selection import train_test_split

(X_train, Y_train), (X_test, Y_test) =  keras.datasets.fashion_mnist.load_data()
X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=.4, random_state=42, stratify=Y_test)
Y_train = keras.utils.to_categorical(Y_train, 10)
Y_test = keras.utils.to_categorical(Y_test, 10)
Y_val = keras.utils.to_categorical(Y_val, 10)


```

### Normalizacion


## Entrenamiento

## Evaluacion

