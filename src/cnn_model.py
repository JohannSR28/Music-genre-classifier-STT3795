#Inspo:
# https://victorzhou.com/blog/keras-cnn-tutorial/
# https://blent.ai/blog/a/cnn-comment-ca-marche

# Choix de librairie : https://www.datacamp.com/fr/tutorial/introduction-to-convolutional-neural-networks-cnns

# Choix du nombre de filtres, neurones et taux dropout: https://www.kaggle.com/code/cdeotte/how-to-choose-cnn-architecture-mnist#Experiment-1

import os
import tensorflow as tf

CONFIG = {
    "cnn_input_dir": "data/cnn_input",
    "batch_size":32,                # Nous allons traiter 32 iages à la fois
    "img_size":224,
    "pool_size":2,
    "epochs":8,                     # Pour le cas de 8 genres musicaux
    "num_classes":8
}

# Prétraitement des images (normalisation des pixels entre 0 et 1, et proportion des images réservé à la validation)
train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Chargement des images pour l'entraînement 
train_generator = train_data_gen.flow_from_directory(
    CONFIG["cnn_input_dir"],
    target_size=(CONFIG["img_size"],CONFIG["img_size"]),
    batch_size=CONFIG["batch_size"],
    class_mode = "categorical",
    subset="training"
)

# Chargement des images pour la validation
validation_generator = train_data_gen.flow_from_directory(
    CONFIG["cnn_input_dir"],
    target_size=(CONFIG["img_size"],CONFIG["img_size"]),
    batch_size=CONFIG["batch_size"],
    class_mode = "categorical",
    subset="validation"
)

# Modèle CNN
model = tf.keras.models.Sequential([

    # COUCHE CONVOLUTION ET ACTIVATION (1ère)
    # Nous débutons avec 32 filtres (3,3) et activons seulement les valeurs positives (pour accélérer la convergence)
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(CONFIG["img_size"],CONFIG["img_size"], 3)),
    
    tf.keras.layers.BatchNormalization(),

    # COUCHE POOLING (1ère)
    # Réduction de la taille de l'image de moitié
    tf.keras.layers.MaxPooling2D(pool_size=(CONFIG["pool_size"], CONFIG["pool_size"])),

    # COUCHE CONVOLUTION ET ACTIVATION (2ième)
    # Augmentation à 64 filtres pour apprendre des motifs plus complexes
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    
    tf.keras.layers.BatchNormalization(),

    # COUCHE POOLING (2ième)
    # Réduction de la taille de l'image de moitié
    tf.keras.layers.MaxPooling2D(pool_size=(CONFIG["pool_size"], CONFIG["pool_size"])),

    # COUCHE CONVOLUTION ET ACTIVATION (3ième)
    # Augmentation à 64 filtres pour apprendre des motifs plus complexes
    tf.keras.layers.Conv2D(128, (3,3), activation = 'relu'),
    
    tf.keras.layers.BatchNormalization(),

    # COUCHE POOLING (3ième)
    # Réduction de la taille de l'image de moitié
    tf.keras.layers.MaxPooling2D(pool_size=(CONFIG["pool_size"], CONFIG["pool_size"])),

    # COUCHE FLATTEN 
    # Passage de 2D à 1D
    tf.keras.layers.Flatten(),

    # COUCHE FULLY CONNECTED
    tf.keras.layers.Dense(256, activation = 'relu'),#128 neurones
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(CONFIG["num_classes"], activation = "softmax")
])

# Compilation du modèle
model.compile(
    optimizer = tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entraînement du modèle
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=CONFIG["epochs"]
)

model.save('cnn.keras')