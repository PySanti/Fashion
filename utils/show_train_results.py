import matplotlib.pyplot as plt

def show_train_results(history):
    """
    Genera dos gráficas a partir del historial de entrenamiento de Keras:
    1. Loss vs Val_loss
    2. Accuracy vs Val_accuracy
    
    Args:
        history: Objeto retornado por model.fit() en Keras/TensorFlow.
    """
    # Obtener las métricas del historial
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)
    
    # Configurar el estilo de las gráficas
    plt.figure(figsize=(12, 5))
    
    # Gráfica 1: Loss vs Val_loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Gráfica 2: Accuracy vs Val_accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, val_accuracy, 'r-', label='Validation Accuracy', linewidth=2)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ajustar layout y mostrar
    plt.tight_layout()
    plt.show()
