import librosa
import librosa.display
import numpy as np
import os
from tqdm import tqdm 
import matplotlib.pyplot as plt
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


# Fonction créant une image de spectrogramme à partir d'un fichier audio du
# dossier fma_small pour ensuite la sauvegarder
def save_spectrogram(audio_path, output_path, duration=30):
    try:
        y,sr = librosa.load(audio_path, duration=duration, sr=None) # Signal audio et fréquence d'échantillonnage 
        S = librosa.feature.melspectrogram(y=y, sr=sr) # Spectrogram en Mel
        S_db = librosa.power_to_db(S, ref=np.max) # Conversion en décibels

        # Création de l'image
        #plt.figure(figsize=(3,3))
        plt.figure(figsize=(2.24,2.24))
        librosa.display.specshow(S_db, sr=sr)
        plt.axis('off')
        plt.tight_layout()
        
        # Sauvegarder l"image
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        return True
    
    except Exception as e:
        print(f"Erreur {e} avec {audio_path}")
        return False

# Fonction qui parcourt tous les fichiers mp3 de fml_small et génère 
# leurs spectrogrammes    
def generate_spectrograms():  

    # Initialisation du tableau qui contiendra les chemins de tous les fichiers .mp3
    all_mp3_files_paths = []

    # Parcours tous les fichiers pour créer un tableau de chemins des fichiers .mp3
    for root, dirs, files in os.walk(CONFIG["audio_dir"]):
        for file in files:
            # Au cas où un fichier n'est pas mp3
            if file.endswith('.mp3'):
                cur_path = os.path.join(root,file)
                all_mp3_files_paths.append(cur_path)

    # Parcours tous les fichiers pour générer les spectrogrammes en affichant la barre de progression
    for audio_path in tqdm(all_mp3_files_paths, desc="Génération des spectrogrammes"):            
            # Extraction du nom du fichier
            track_id = os.path.splitext(os.path.basename(audio_path))[0]
            # Chemin du fichier de sortie
            output_path = os.path.join(CONFIG["spect_output_dir"], f"{track_id}.png")

            # Pour éviter doublons
            if not os.path.exists(output_path):
                save_spectrogram(audio_path,output_path)


if __name__ == "__main__":
    generate_spectrograms()



# v1
#Génération des spectrogrammes:   7%|███▋                                                 | 550/8000 [28:15<6:22:52,  3.08s/it]
# v2
#Génération des spectrogrammes:   6%|█▊                           | 500/8000 [12:54<3:13:42,  1.55s/it]^C