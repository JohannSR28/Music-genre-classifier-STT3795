from random_forest import CONFIG, extract_features
import joblib
import numpy as np



def predict_new_audio(audio_path):
    # Charger le modèle
    loaded_data = joblib.load(CONFIG["model_path"])
    model = loaded_data['model']
    le = loaded_data['label_encoder']
    
    # Extraire les features
    features = extract_features(audio_path)
    
    # Prédiction
    proba = model.predict_proba([features])
    genre_idx = np.argmax(proba)
    return le.inverse_transform([genre_idx])[0]


