import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report

CONFIG = {
    "cnn_input_dir": "data/cnn_input",
    "batch_size":32,                
    "img_size":224,                    
    "num_classes":8,
    "model_path": "cnn.keras"
}

# Recharger model
model = tf.keras.models.load_model('cnn.keras')

# Prétraitement des images pour la validation (normalisation des pixels entre 0 et 1, et proportion des images réservé à la validation)
data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Chargement des images pour la validation
validation_generator = data_gen.flow_from_directory(
    CONFIG["cnn_input_dir"],
    target_size=(CONFIG["img_size"],CONFIG["img_size"]),
    batch_size=CONFIG["batch_size"],
    class_mode = "categorical",
    shuffle = False,
    subset="validation"
)

# PRÉDICTIONS
# Probabilité pour chaque classes à
predicted_proba = model.predict(validation_generator, verbose=1)
# Classes choisies par le modèle
predicted_classes = np.argmax(predicted_proba, axis=1)
# Les vraies étiquettes
true_classes = validation_generator.classes

# Faire correspondre les indexes aux genres musicaux
class_idx = validation_generator.class_indices
idx_to_class = {v: k for k, v in class_idx.items()}

# Rapport
report=classification_report(
    true_classes,
    predicted_classes,
    target_names=[idx_to_class[i] for i in range(len(idx_to_class))],
    digits=2
)

# Afichage du rapport
print("\nÉvaluation du modèle CNN :\n")
print(report)