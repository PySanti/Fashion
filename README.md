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

### Conversion de targets

Teniendo en cuenta que es un problema de clasificacion multinomial, las neuronas de la output layer deberan implementar `softmax` como funcion de activacion, para ello, debemos modificar el formato de los targets del conjunto, usando el metodo `.to_categorical()`.

### Normalizacion


## Entrenamiento

## Evaluacion

