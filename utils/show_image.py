
import matplotlib.pyplot as plt
import numpy as np


def show_image(imagen : np.ndarray):
    """
    Genera una ventana gráfica utilizando Matplotlib para mostrar una imagen
    del conjunto de datos Fashion MNIST.

    Args:
        imagen (np.ndarray): Un arreglo NumPy que representa la imagen a mostrar.
                             Debe tener la forma (28, 28) o similar,
                             típicamente en escala de grises.
    """
    plt.figure(figsize=(2, 2)) # Ajusta el tamaño de la figura para una imagen de 28x28
    plt.imshow(imagen, cmap='gray') # Muestra la imagen en escala de grises
    plt.axis('off') # Oculta los ejes para una visualización más limpia
    plt.title("Imagen Fashion MNIST") # Añade un título
    plt.show() # Muestra la ventana gráfica


