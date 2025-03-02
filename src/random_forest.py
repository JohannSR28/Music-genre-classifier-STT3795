import librosa
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import time
from pydub.utils import mediainfo



# Configuration globale
CONFIG = {
    "audio_dir": "data/raw/fma_small",
    "metadata_path": "data/raw/fma_metadata/tracks.csv",
    "feature_params": {
        "n_mfcc": 20,
        "n_fft": 2048,
        "hop_length": 512,
        "duration": 30,
        "spectral_contrast_bands": 7,  
        "tempogram_win_length": 100     
    },
    "model_path": "models/random_forest.joblib",
    "test_size": 0.2,
    "random_state": 42
}

def load_metadata():
    """Charge les métadonnées et extrait les genres principaux"""
    tracks = pd.read_csv(CONFIG["metadata_path"], index_col=0, header=[0, 1])
    return tracks['track', 'genre_top']

def extract_features(audio_path):
    """Extrait les caractéristiques audio avec gestion d'erreurs"""
    try:
        audio, sr = librosa.load(
            audio_path, 
            duration=CONFIG["feature_params"]["duration"],
            sr=None  # Conserve la fréquence d'origine
        )
        
        # Extraction des caractéristiques
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=CONFIG["feature_params"]["n_mfcc"],
            n_fft=CONFIG["feature_params"]["n_fft"],
            hop_length=CONFIG["feature_params"]["hop_length"]
        )
        
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=sr,
            n_fft=CONFIG["feature_params"]["n_fft"],
            hop_length=CONFIG["feature_params"]["hop_length"]
        )

        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=sr,
            n_bands=CONFIG["feature_params"]["spectral_contrast_bands"]
        )
        
        tempogram = librosa.feature.tempogram(
            y=audio,
            sr=sr,
            win_length=CONFIG["feature_params"]["tempogram_win_length"]
        )
        
       # Agrégation statistique 
        features = np.concatenate([
            # MFCC (20 moyennes + 20 écarts-types)
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
            
            # Chroma (12 moyennes + 12 écarts-types)
            chroma.mean(axis=1),
            chroma.std(axis=1),
            
            # Spectral Contrast (7 bands × 2 stats)
            spectral_contrast.mean(axis=1),
            spectral_contrast.std(axis=1),
            
            # Tempogram (reduced to 50 coefficients × 2 stats)
            tempogram.mean(axis=1)[:50],  # On garde les 50 premiers coefficients
            tempogram.std(axis=1)[:50]
        ])
        
        return features #178 caratéristiques
    
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None


def validate_audio(path):
    """Vérifie la validité d'un fichier audio"""
    try:
        info = mediainfo(path)
        return float(info.get('duration', 0)) >= 27  # Filtre les fichiers tronqués
    except:
        return False
    
def process_single_file(root, file):
    audio_path = os.path.join(root, file)
    if not validate_audio(audio_path):
        return None  # Ignore les fichiers invalides
    
    return extract_features(audio_path)
    
def create_dataset(genres):
    """Crée le dataset complet avec features et labels"""
    features = []
    track_ids = []
    
    # Parcours récursif avec barre de progression
    for root, _, files in tqdm(os.walk(CONFIG["audio_dir"]), desc="Processing files"):
        for file in files:
            if file.endswith('.mp3'):

                track_id = int(file.split('.')[0])
                feat = process_single_file(root, file)

                if feat is not None:
                    features.append(feat)
                    track_ids.append(track_id)
    
    # Création du DataFrame
    df = pd.DataFrame(features, index=track_ids)
    df['genre'] = df.index.map(genres) 
    df = df.dropna()

    # Sauvegarde du DataFrame dans un fichier CSV
    df.to_csv("data/processed/randomForestfeatures.csv", index=True)
    print("Dataset saved to data/processed/randomForestfeatures.csv")
    
    return df

def train_model(df):
    """Entraîne et évalue le modèle de Forêt Aléatoire"""
    # Encodage des labels
    le = LabelEncoder()
    y = le.fit_transform(df['genre'])
    
    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop('genre', axis=1),
        y,
        test_size=CONFIG["test_size"],
        stratify=y,
        random_state=CONFIG["random_state"]
    )
    
    # Initialisation et entraînement du modèle
    model = RandomForestClassifier(
        n_estimators=500,        
        max_depth=30,             
        max_features='sqrt',      
        min_samples_split=20,     
        class_weight='balanced',
        n_jobs=-1,
        random_state=CONFIG["random_state"]
    )
    
    model.fit(X_train, y_train)
    
    # Évaluation
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    return model, le

def train_sub_model(df):
    """Entraîne et évalue le modèle de Forêt Aléatoire"""
    # Encodage des labels
    le = LabelEncoder()
    y = le.fit_transform(df['genre'])
    
    # Sélection des 64 premières caractéristiques
    X = df.drop('genre', axis=1).iloc[:, :64]  # Sélectionne les 64 premières colonnes
    
    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,  # Utilise seulement les 64 premières features
        y,
        test_size=CONFIG["test_size"],
        stratify=y,
        random_state=CONFIG["random_state"]
    )
    
    # Initialisation et entraînement du modèle
    model = RandomForestClassifier(
        n_estimators=500,        
        max_depth=30,             
        max_features='sqrt',      
        min_samples_split=20,     
        class_weight='balanced',
        n_jobs=-1,
        random_state=CONFIG["random_state"]
    )
    
    model.fit(X_train, y_train)
    
    # Évaluation
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    return model, le


def main():
    # Chargement des métadonnées
    genres = load_metadata()

    # Mesure du temps d'extraction des features
    start_time = time.time()
    # df = create_dataset(genres)
    df = pd.read_csv("data/processed/randomForestfeatures.csv")    
    extraction_time = time.time() - start_time
    print(f"Temps d'extraction des features : {extraction_time:.2f} secondes\n")

    # Entraînement du modèle complet (avec toutes les features)
    print("Entraînement du modèle complet (toutes les features) :")
    start_time = time.time()
    model_full, label_encoder_full = train_model(df)
    training_time_full = time.time() - start_time
    print(f"Temps d'entraînement (modèle complet) : {training_time_full:.2f} secondes\n")

    # Entraînement du modèle sur sous-ensemble (64 premières features)
    print("Entraînement du modèle avec les 64 premières features :")
    start_time = time.time()
    model_sub, label_encoder_sub = train_sub_model(df)
    training_time_sub = time.time() - start_time
    print(f"Temps d'entraînement (modèle sous-ensemble) : {training_time_sub:.2f} secondes\n")

    # Sauvegarde des artefacts
    os.makedirs(os.path.dirname(CONFIG["model_path"]), exist_ok=True)
    joblib.dump({
        'model_full': model_full,
        'label_encoder_full': label_encoder_full,
        'features_full': df.drop('genre', axis=1).columns.tolist(),
        'model_sub': model_sub,
        'label_encoder_sub': label_encoder_sub,
        'features_sub': df.drop('genre', axis=1).iloc[:, :64].columns.tolist()
    }, CONFIG["model_path"])

    print(f"Les modèles ont été sauvegardés dans {CONFIG['model_path']}")

if __name__ == "__main__":
    main()
