import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
from tqdm import tqdm
import joblib
import time
import os

# Configuration globale
CONFIG = {
    "feature_csv": "data/processed/randomForestfeatures.csv",
    "model_path": "models/rf_improved.joblib",
    "test_size": 0.2,
    "random_state": 42
}

def load_features():
    """Charge les features préextraites du CSV"""
    print(f"Chargement des features depuis {CONFIG['feature_csv']}...")
    df = pd.read_csv(CONFIG["feature_csv"])
    
    # Si la première colonne est l'index, on la renomme
    if 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'track_id'}, inplace=True)
    
    print(f"Dataset chargé avec {df.shape[0]} échantillons et {df.shape[1]} colonnes")
    return df

def analyze_feature_importance(model, feature_names, top_n=20):
    """Visualise et retourne les features les plus importantes"""
    # Dans un Random Forest, l'importance des features est calculée en mesurant 
    # la diminution de l'impureté (Gini ou entropie) lors des splits utilisant
    # cette feature. Une feature est importante si elle contribue à réduire 
    # significativement l'impureté dans les nœuds de l'arbre.
    feature_importance = model.feature_importances_
    
    # Création d'un DataFrame pour faciliter le tri
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    
    # Tri par importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Explications sur les 5 premières features
    print("=== Explication des caractéristiques les plus importantes ===")
    print("Les scores d'importance indiquent à quel point chaque feature contribue à la diminution de l'impureté")
    print("dans les arbres de la forêt. Plus le score est élevé, plus la feature aide à discriminer entre les genres.\n")
    
    for i, (feature, importance) in enumerate(zip(importance_df['Feature'][:5], importance_df['Importance'][:5])):
        print(f"{i+1}. {feature}: {importance:.4f} - Cette caractéristique contribue à {importance*100:.2f}% de la puissance discriminante du modèle")
    
    # Visualisation
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'][:top_n], importance_df['Importance'][:top_n])
    plt.xlabel('Importance (réduction de l\'impureté Gini)')
    plt.title(f'Top {top_n} Most Important Features')
    plt.gca().invert_yaxis()  # Pour avoir la plus importante en haut
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()
    
    return importance_df

def plot_confusion_matrix(y_true, y_pred, labels):
    """Affiche la matrice de confusion avec explications"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels,
                yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Calcul et affichage des taux de succès et de confusion par genre
    print("\n=== Analyse de la matrice de confusion ===")
    print("Comment lire la matrice de confusion:")
    print("- Les lignes représentent les genres réels")
    print("- Les colonnes représentent les genres prédits")
    print("- La diagonale (coin supérieur gauche à coin inférieur droit) montre les prédictions correctes")
    print("- Les valeurs hors diagonale sont des erreurs de classification\n")
    
    total_per_genre = np.sum(cm, axis=1)
    
    # Taux de succès par genre (precisión)
    print("Taux de succès par genre:")
    for i, genre in enumerate(labels):
        accuracy = cm[i, i] / total_per_genre[i] * 100
        print(f"{genre}: {accuracy:.2f}% ({cm[i, i]}/{total_per_genre[i]})")
    
    # Analyse des principales confusions
    print("\nPrincipales confusions:")
    confusions = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                confusion_rate = cm[i, j] / total_per_genre[i] * 100
                confusions.append((labels[i], labels[j], cm[i, j], confusion_rate))
    
    # Tri par nombre d'échantillons confondus
    confusions.sort(key=lambda x: x[2], reverse=True)
    
    # Affiche les 5 confusions les plus fréquentes
    for i in range(min(5, len(confusions))):
        real, pred, count, rate = confusions[i]
        print(f"{real} classifié comme {pred}: {count} fois ({rate:.2f}%)")
    
    return cm

def train_baseline_model(df):
    """Entraîne et évalue le modèle de base avec normalisation dès le début"""
    # Séparation features/target
    if 'track_id' in df.columns:
        X = df.drop(['genre', 'track_id'], axis=1)
    else:
        X = df.drop(['genre'], axis=1)
    
    # Encodage des labels
    le = LabelEncoder()
    y = le.fit_transform(df['genre'])
    
    # Normalisation des données - Ajoutée dès le début
    print("Normalisation des données...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Séparation train/test avec les données normalisées
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y, test_size=CONFIG["test_size"], 
        stratify=y, random_state=CONFIG["random_state"]
    )
    
    # Initialisation et entraînement du modèle
    print("Entraînement du modèle de base...")
    start_time = time.time()
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
    training_time = time.time() - start_time
    
    # Évaluation
    print("Évaluation du modèle de base...")
    y_pred = model.predict(X_test)
    print("=== Modèle de Base (avec normalisation) ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print(f"Temps d'entraînement: {training_time:.2f} secondes\n")
    
    # Visualisation de la matrice de confusion
    plot_confusion_matrix(y_test, y_pred, le.classes_)
    
    # Analyse de l'importance des features
    importance_df = analyze_feature_importance(model, X.columns)
    
    return model, le, X_train, X_test, y_train, y_test, importance_df, scaler

def optimize_hyperparameters(X_train, y_train):
    """Optimise les hyperparamètres avec GridSearchCV et tqdm"""
    print("Optimisation des hyperparamètres...")
    
    param_grid = {
        'n_estimators': [300, 500, 700],
        'max_depth': [20, 30, 40],
        'min_samples_split': [10, 20, 30],
        'max_features': ['sqrt', 'log2'],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestClassifier(class_weight='balanced', random_state=CONFIG["random_state"])
    
    # Calcul du nombre total d'itérations pour tqdm
    param_combinations = 1
    for param_values in param_grid.values():
        param_combinations *= len(param_values)
    total_iterations = param_combinations * 3  # 3 pour le CV à 3 folds
    
    print(f"Évaluation de {param_combinations} combinaisons de paramètres avec 3-fold CV ({total_iterations} itérations)")
    
    # Instancier GridSearchCV avec verbose=2 pour voir la progression
    grid_search = GridSearchCV(
        rf, param_grid, cv=3, scoring='f1_macro', 
        n_jobs=-1, verbose=2
    )
    
    # Utiliser un wrapper pour afficher une barre de progression
    with tqdm(total=total_iterations, desc="GridSearchCV Progress") as pbar:
        # Division par lots pour mise à jour de la barre de progression
        batch_size = max(1, total_iterations // 100)
        completed = 0
        
        # Ajuster X_train et y_train pour les lots
        grid_search.fit(X_train, y_train)
        pbar.update(total_iterations - completed)  # Update any remaining iterations
    
    print(f"Meilleurs paramètres: {grid_search.best_params_}")
    print(f"Meilleur score F1: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def feature_selection(X_train, y_train, X_test, improved_model):
    """Sélection automatique des caractéristiques les plus importantes avec explications"""
    print("\n=== Sélection des caractéristiques ===")
    print("Critère de sélection: Une caractéristique est conservée si son importance")
    print("est supérieure à la moyenne des importances de toutes les caractéristiques.")
    
    selector = SelectFromModel(improved_model, threshold='mean')
    
    print("Application de la sélection de caractéristiques...")
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Obtenir les indices des caractéristiques sélectionnées
    selected_indices = selector.get_support(indices=True)
    selected_features = X_train.columns[selected_indices]
    
    print(f"\nNombre de caractéristiques originales: {X_train.shape[1]}")
    print(f"Nombre de caractéristiques sélectionnées: {X_train_selected.shape[1]} ({X_train_selected.shape[1]/X_train.shape[1]*100:.1f}%)")
    
    # Afficher les importances des caractéristiques sélectionnées
    importance_values = improved_model.feature_importances_[selected_indices]
    importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': importance_values
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 caractéristiques sélectionnées par importance:")
    for i, (feature, importance) in enumerate(zip(importance_df['Feature'][:10], importance_df['Importance'][:10])):
        print(f"{i+1}. {feature}: {importance:.4f}")
    
    return X_train_selected, X_test_selected, selected_features, importance_df

def analyze_errors(X_test, y_test, y_pred, le):
    """Analyse des échantillons mal classés"""
    print("\n=== Analyse des erreurs de classification ===")
    misclassified_indices = np.where(y_test != y_pred)[0]
    misclassified_genres = le.inverse_transform(y_test[misclassified_indices])
    predicted_genres = le.inverse_transform(y_pred[misclassified_indices])
    
    error_analysis = pd.DataFrame({
        'True Genre': misclassified_genres,
        'Predicted Genre': predicted_genres
    })
    
    error_counts = error_analysis.groupby(['True Genre', 'Predicted Genre']).size().reset_index(name='Count')
    error_counts = error_counts.sort_values('Count', ascending=False)
    
    print(f"Nombre total d'erreurs: {len(misclassified_indices)} sur {len(y_test)} échantillons de test ({len(misclassified_indices)/len(y_test)*100:.2f}%)")
    
    print("\nPrincipales erreurs de classification:")
    for i, row in error_counts.head(10).iterrows():
        print(f"{row['True Genre']} classifié comme {row['Predicted Genre']}: {row['Count']} fois")
    
    return error_analysis

def categorize_features(importance_df, top_n=30):
    """
    Catégorise les caractéristiques les plus importantes selon leur type et statistique
    
    Args:
        importance_df: DataFrame contenant les caractéristiques et leur importance
        top_n: Nombre de caractéristiques importantes à analyser
    
    Returns:
        Un DataFrame avec les caractéristiques catégorisées
    """
    # Définir les plages d'indices pour chaque catégorie
    categories = {
        'MFCC - moyenne': (0, 19),
        'MFCC - écart-type': (20, 39),
        'Chroma - moyenne': (40, 51),
        'Chroma - écart-type': (52, 63),
        'Spectral Contrast - moyenne': (64, 70),
        'Spectral Contrast - écart-type': (71, 77),
        'Tempogram - moyenne': (78, 127),
        'Tempogram - écart-type': (128, 177)
    }
    
    # Fonction pour déterminer la catégorie d'une caractéristique
    def get_category(feature_index):
        feature_idx = int(feature_index)
        for category, (start, end) in categories.items():
            if start <= feature_idx <= end:
                return category
        return "Inconnu"
    
    # Prendre les top_n caractéristiques
    top_features = importance_df.head(top_n).copy()
    
    # Ajouter la catégorie
    top_features['Catégorie'] = top_features['Feature'].apply(get_category)
    
    return top_features

def analyze_feature_categories(categorized_df):
    """
    Analyse la distribution des catégories parmi les caractéristiques importantes
    
    Args:
        categorized_df: DataFrame avec les caractéristiques catégorisées
    """
    # Compter les occurrences de chaque catégorie
    category_counts = categorized_df['Catégorie'].value_counts()
    
    # Calculer le pourcentage
    category_percentage = (category_counts / len(categorized_df) * 100).round(1)
    
    # Créer un DataFrame des résultats
    results = pd.DataFrame({
        'Nombre': category_counts,
        'Pourcentage (%)': category_percentage
    })
    
    print("\n=== Distribution des catégories de caractéristiques importantes ===")
    print(results)
    
    # Visualiser la distribution
    plt.figure(figsize=(12, 6))
    plt.bar(results.index, results['Pourcentage (%)'])
    plt.xlabel('Catégorie')
    plt.ylabel('Pourcentage (%)')
    plt.title('Distribution des catégories parmi les caractéristiques importantes')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('feature_categories.png')
    plt.show()
    
    return results

def main():
    # Charger les features
    df = load_features()
    
    # Entraîner et évaluer le modèle de base (avec normalisation dès le début)
    model, le, X_train, X_test, y_train, y_test, importance_df, scaler = train_baseline_model(df)
    
    # Optimiser les hyperparamètres
    print("\n" + "="*50)
    print("Phase d'optimisation des hyperparamètres")
    print("="*50)
    best_model, best_params = optimize_hyperparameters(X_train, y_train)
    
    # Entraîner un modèle avec les meilleurs hyperparamètres
    print("\n" + "="*50)
    print("Évaluation du modèle avec meilleurs hyperparamètres")
    print("="*50)
    print(f"Entraînement avec paramètres optimisés: {best_params}")
    
    with tqdm(total=100, desc="Training Optimized Model") as pbar:
        best_model.fit(X_train, y_train)
        pbar.update(100)
    
    print("Prédiction et évaluation...")
    y_pred_best = best_model.predict(X_test)
    print("\n=== Modèle avec hyperparamètres optimisés ===")
    print(classification_report(y_test, y_pred_best, target_names=le.classes_))
    
    # Visualisation de la matrice de confusion du modèle optimisé
    plot_confusion_matrix(y_test, y_pred_best, le.classes_)
    
    # Sélection des caractéristiques les plus importantes
    X_train_selected, X_test_selected, selected_features, importance_selected_df = feature_selection(X_train, y_train, X_test, best_model)
    
    # Entraîner avec les caractéristiques sélectionnées
    print("\n" + "="*50)
    print("Évaluation du modèle avec caractéristiques sélectionnées")
    print("="*50)
    
    selected_model = RandomForestClassifier(**best_params, class_weight='balanced', random_state=CONFIG["random_state"])
    
    with tqdm(total=100, desc="Training Feature-Selected Model") as pbar:
        selected_model.fit(X_train_selected, y_train)
        pbar.update(100)
    
    print("Prédiction et évaluation...")
    y_pred_selected = selected_model.predict(X_test_selected)
    print("\n=== Modèle avec caractéristiques sélectionnées ===")
    print(classification_report(y_test, y_pred_selected, target_names=le.classes_))
    
    # Analyser les erreurs
    error_analysis = analyze_errors(X_test, y_test, y_pred_selected, le)
    
    # AJOUT: Analyser les catégories des caractéristiques importantes pour le modèle final
    print("\n" + "="*50)
    print("Analyse des catégories de caractéristiques")
    print("="*50)
    
    # Extraire les importances du modèle final
    selected_feature_importance = selected_model.feature_importances_
    
    # Créer un DataFrame avec les caractéristiques sélectionnées et leurs importances
    final_importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': selected_feature_importance
    }).sort_values('Importance', ascending=False)
    
    # Catégoriser les 30 caractéristiques les plus importantes
    categorized_features = categorize_features(final_importance_df, top_n=30)
    
    # Analyser et visualiser la distribution des catégories
    category_distribution = analyze_feature_categories(categorized_features)
    
    # Afficher les 30 caractéristiques les plus importantes avec leur catégorie
    print("\n=== 30 caractéristiques les plus importantes avec leur catégorie ===")
    for i, (_, row) in enumerate(categorized_features.iterrows(), 1):
        print(f"{i}. {row['Feature']}: {row['Importance']:.4f} - {row['Catégorie']}")
    
    # Corriger la partie validation croisée
    print("\nRéalisation d'une validation croisée sur le modèle final...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=CONFIG["random_state"])

    with tqdm(total=5, desc="Cross-Validation") as pbar:
        scores = []
        # Convertir X_train_selected en DataFrame si ce n'est pas déjà le cas
        if isinstance(X_train_selected, np.ndarray):
            X_train_selected_df = pd.DataFrame(X_train_selected)
        else:
            X_train_selected_df = X_train_selected
            
        for train_idx, val_idx in cv.split(X_train_selected_df, y_train):
            # Utiliser directement les indices avec les tableaux NumPy
            X_cv_train, X_cv_val = X_train_selected[train_idx], X_train_selected[val_idx]
            y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
            
            cv_model = RandomForestClassifier(**best_params, class_weight='balanced', random_state=CONFIG["random_state"])
            cv_model.fit(X_cv_train, y_cv_train)
            y_cv_pred = cv_model.predict(X_cv_val)
            
            # Calculer le F1-score macro moyen
            f1 = f1_score(y_cv_val, y_cv_pred, average='macro')
            scores.append(f1)
            
            pbar.update(1)
    
    print(f"F1-score moyen (validation croisée): {np.mean(scores):.3f} (+/- {np.std(scores) * 2:.3f})")
    
    # Sauvegarder le modèle amélioré
    os.makedirs(os.path.dirname(CONFIG["model_path"]), exist_ok=True)
    joblib.dump({
        'model': selected_model,
        'scaler': scaler,
        'label_encoder': le,
        'selected_features': selected_features.tolist(),
        'best_params': best_params,
        'feature_categories': categorized_features.to_dict()  # Ajouter les catégories au modèle sauvegardé
    }, CONFIG["model_path"])
    
    print(f"\nLe modèle amélioré a été sauvegardé dans {CONFIG['model_path']}")

if __name__ == "__main__":
    main()