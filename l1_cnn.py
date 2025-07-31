import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar100
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import random

tf.random.set_seed(0)  # Reproduzierbarkeit

# Aufgabe 1
# Klassennamen aus der Meta-Datei laden und als Dictionary speichern
def load_dict(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='utf-8')
    return dict

# Laden der CIFAR-100 Daten
def load_cifar100():    
    (train_data, train_labels), (test_data, test_labels) = cifar100.load_data(label_mode='fine') # CIFAR-100 laden
    class_names = load_dict('meta')['fine_label_names'] # Laden der Klassennamen
    return train_data, train_labels, test_data, test_labels, class_names

# zufällig 15 Bilder in 5x3 anzeigen
def create_grid(images, labels, class_names):
    fig, axs = plt.subplots(3, 5, figsize=(15, 9))
    for i in range(3):
        for j in range(5):
            index = random.randint(0, len(images) - 1)
            image = images[index]
            label = class_names[int(labels[index])] + image.shape.__str__()  # Klassennamen und Bildgröße
            axs[i, j].imshow(image)
            axs[i, j].set_title(label, fontsize=8)
            axs[i, j].axis('off')
    plt.tight_layout()
    plt.show()


# Aufgabe 2
def create_cnn_without_regularization():
    # Modell initialisieren
    model = keras.Sequential()

    # Layer 1
    model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same')) # Alle Bilder von CIFAR-100 haben die Eingabeform: 32x32 Pixel, 3 Farbkanäle (RGB)
    model.add(layers.MaxPooling2D((2, 2)))

    # Layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Layer 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Klassifikation
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(100, activation='softmax'))  # 100 Klassen
    
    # Kompilieren
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    model.summary()  # Modellübersicht anzeigen
    return model


# Aufgabe 3
## Daten Normalisieren & Splitten
def train_cnn(model, train_data, test_data, train_labels):
    train_data = train_data.astype('float32') / 255.0
    test_data = test_data.astype('float32') / 255.0

    # Training mit 20% Validierung
    history = model.fit(train_data, train_labels,
                        epochs=50,
                        validation_split=0.2,
                        batch_size=64)

    ## Visualisierung der Genauigkeit
    plt.plot(history.history['val_accuracy'], label='Validierungsgenauigkeit')
    #plt.plot(history.history['accuracy'], label='Trainingsgenauigkeit')
    plt.xlabel('Epoche')
    plt.ylabel('Genauigkeit')
    plt.legend()
    plt.grid(True)
    plt.title('Genauigkeit mit L1 Regularisierung')
    plt.show()


# Aufgabe 4
## Regulierung mit Dropout
def create_cnn_dropout_regularization():
    # Modell initialisieren
    model = keras.Sequential()

    # Layer 1
    model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same')) # Alle Bilder von CIFAR-100 haben die Eingabeform: 32x32 Pixel, 3 Farbkanäle (RGB)
    model.add(layers.MaxPooling2D((2, 2)))

    # Layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Layer 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Klassifikation
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(100, activation='softmax'))  # 100 Klassen
    
    # Kompilieren
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    model.summary()  # Modellübersicht anzeigen
    return model


## Regulierung mit Data Augmentation
def create_cnn_l1_regularization():
    # Modell initialisieren
    model = keras.Sequential()

    # Layer 1
    model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same')) # Alle Bilder von CIFAR-100 haben die Eingabeform: 32x32 Pixel, 3 Farbkanäle (RGB)
    model.add(layers.MaxPooling2D((2, 2)))

    # Layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Layer 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Klassifikation
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu', kernel_regularizer='l1'))
    model.add(layers.Dense(100, activation='softmax'))  # 100 Klassen
    
    # Kompilieren
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    model.summary()  # Modellübersicht anzeigen
    return model

train_data, train_labels, test_data, test_labels, class_names = load_cifar100()
# create_grid(train_data, train_labels, class_names)

# Erstellen und Trainieren des CNN ohne Regularisierung
# model = create_cnn_without_regularization()
# train_cnn(model, train_data, test_data, train_labels)

# Erstellen und Trainieren des CNN mit Dropout Regularisierung
# model_dropout = create_cnn_dropout_regularization()
# train_cnn(model_dropout, train_data, test_data, train_labels)

# Erstellen und Trainieren des CNN mit L1 Regularisierung
model_l1 = create_cnn_l1_regularization()
train_cnn(model_l1, train_data, test_data, train_labels)
