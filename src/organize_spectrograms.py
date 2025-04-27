import os
from tqdm import tqdm 
import pandas as pd
import shutil

# Configuration globale
CONFIG={
    "audio_dir": "data/raw/fma_small",
    "metadata_path": "data/raw/fma_metadata/tracks.csv",
    "spect_output_dir" :"data/processed/spectrograms_v2", #"spect_output_dir" :"data/processed/spectrograms",
    "duration":30,
    "cnn_input_dir": "data/cnn_input"

}

def load_metadata():
    """Charge les métadonnées et extrait les genres principaux"""
    tracks = pd.read_csv(CONFIG["metadata_path"], index_col=0, header=[0, 1])
    return tracks['track', 'genre_top']

# Organise les spectrogrammes selon leur genre
def organize_spectrograms(genres):
    spectrogram_files = [f for f in os.listdir(CONFIG["spect_output_dir"])]

    # Parcours tous les fichiers de spectrogrammes pour les organiser en affichant la barre de progression
    for file_name in tqdm(spectrogram_files, desc="Organisation des spectrogrammes"):

        # Extraction du nom du fichier courant
        track_id = int(file_name.split('.')[0])
        # Extraction du genre du fichier courant
        genre = genres.get(track_id)

        # Si le genre n'est pas connu, on ignore ce fichier
        if pd.isna(genre):
            continue

        # Création du dossier de genre
        genre_folder = os.path.join(CONFIG["cnn_input_dir"], genre)
        os.makedirs(genre_folder, exist_ok=True)

        # Chemins source et destination
        source_path=os.path.join(CONFIG["spect_output_dir"],file_name)
        destination_path = os.path.join(genre_folder, file_name)

        # Faire une copie du spectrogramme dans son dossier de destination
        shutil.copy(source_path, destination_path) 

if __name__ == "__main__":
    genres = load_metadata()
    organize_spectrograms(genres)

# Organisation des spectrogrammes: 100%|█████████████| 499/499 [30:38<00:00,  3.68s/it]