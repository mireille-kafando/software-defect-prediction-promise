"""
Pipeline multimodal inter-projets (CPDP) : Hybride vs Full Pretrained.

Transcription fidèle du dernier bloc du document de travail « import torch » (famille E du tri des
scripts), complété par les fonctions communes que son autrice indique « identiques à celles du code
précédent » (alignement, prétraitement, SMOTE, poids de classes, évaluation). Ces fonctions sont
reprises telles quelles du notebook `05_multimodal/05_hybride_fullpretrained_wpdp.ipynb` (cellule 20),
qui les contient à l'identique.

Seules modifications par rapport à la source : lignes coupées par la mise en page du PDF recollées,
emoji des messages print() perdus lors de l'export PDF.

Modifications de 2026 pour rendre le script exécutable, repérées dans le code par « [2026] ». Elles ne
changent ni le protocole, ni les hyperparamètres, ni les prétraitements :
1. Full Pretrained : ajout de la projection nn.Linear(384, 768) décrite au §4.3.2 du mémoire, avant
   l'encodeur CodeBERT des métriques ; la sortie (768) est ramenée à 384 par la projection
   code_projection existante, pour que la fusion reste 768 = 384 + 384 comme décrit.
2. Synapse ajouté à la liste des projets cibles, comme dans les résultats conservés.
3. Chemins des données et des résultats configurables par variables d'environnement
   (SDP_DATA_DIR, SDP_SOURCE_DIR, SDP_RESULTS_DIR) ; par défaut, les chemins Google Colab d'origine.
La version conservée avant ces modifications reste consultable dans l'historique Git du dépôt.

Protocole : pour chaque projet cible, entraînement sur les versions 1 à n-1 du projet cible et sur
toutes les versions des autres projets, test sur la version n du projet cible.

Points à connaître avant toute réutilisation (voir docs/genealogie_scripts.md) :
- Ce script produit les fichiers training_results_<modèle>_<projet>.pkl dont proviennent les G-Mean
  CPDP du tableau 12 (clés identiques, 11 à 23 époques compatibles avec la patience de 10).
- Dans la version conservée, FullPretrainedModel échouait : la projection des métriques produisait
  384 dimensions alors que CodeBERT attend 768 en entrée (inputs_embeds). La version exécutée pour
  obtenir les résultats Full Pretrained comportait donc une adaptation 384 <-> 768 qui n'a pas été
  conservée ; la modification 1 la rétablit d'après le §4.3.2 du mémoire.
- La liste target_projects de la version conservée ne contenait pas synapse, alors que les résultats
  conservés l'incluent (modification 2).
- Chaque version est standardisée avec son propre StandardScaler (y compris la version de test).
- L'arrêt précoce (patience 10) est piloté par le F1 de la version de test ; la G-Mean rapportée
  est celle de la dernière époque.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from scipy.stats import gmean
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from collections import Counter
import re

# Configuration
class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 25
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-5
    EMBEDDING_DIM = 384
    N_HEADS = 8
    NUM_LAYERS = 2
    DROPOUT = 0.3
    NUM_CLASSES = 2
    # [2026] Chemins configurables ; valeurs par défaut : chemins Google Colab d'origine
    SAVE_PATH = os.path.join(os.environ.get('SDP_RESULTS_DIR', '/content/drive/MyDrive/bug/results/'), '')
    DATA_DIR = os.environ.get('SDP_DATA_DIR', '/content/drive/MyDrive/bug/bug-data/')
    SOURCE_DIR = os.environ.get('SDP_SOURCE_DIR', '/content/drive/MyDrive/bug/bug-data/source-code/source-code/')

    # Paramètres pour l'équilibrage
    LABEL_SMOOTHING = 0.1
    GRADIENT_CLIP = 1.0

    # Paramètres pour l'alignement
    MAX_CODE_LENGTH = 512

    os.makedirs(SAVE_PATH, exist_ok=True)

# Dataset personnalisé avec alignement
class AlignedBugDataset(Dataset):
    def __init__(self, metrics, code_tokens, labels, file_names):
        self.metrics = torch.FloatTensor(metrics)
        self.code_tokens = code_tokens
        self.labels = torch.LongTensor(labels)
        self.file_names = file_names

        assert len(self.metrics) == len(self.labels) == len(self.file_names), f"Mismatch dans les données alignées"
        assert len(self.code_tokens['input_ids']) == len(self.labels), f"Mismatch entre code_tokens et labels"

    def __len__(self):
        return len(self.metrics)

    def __getitem__(self, idx):
        return self.metrics[idx], {
            'input_ids': self.code_tokens['input_ids'][idx],
            'attention_mask': self.code_tokens['attention_mask'][idx]
        }, self.labels[idx], self.file_names[idx]

# STRATÉGIE 1: Modèle hybride (encoder from scratch + decoder pré-entraîné)
class HybridTransBugModel(nn.Module):
    def __init__(self, input_metric_dim=None):
        super().__init__()

        # Projection des métriques vers la dimension d'embedding AVEC LazyLinear
        self.metric_projection = nn.Sequential(
            nn.LazyLinear(Config.EMBEDDING_DIM),
            nn.BatchNorm1d(Config.EMBEDDING_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT)
        )

        # Encoder transformer FROM SCRATCH pour les métriques
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.EMBEDDING_DIM,
            nhead=Config.N_HEADS,
            dropout=Config.DROPOUT,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.metrics_encoder = nn.TransformerEncoder(encoder_layer,
num_layers=Config.NUM_LAYERS)

        # Decoder PRÉ-ENTRAÎNÉ RoBERTa pour le code
        self.code_decoder = RobertaModel.from_pretrained("microsoft/codebert-base")

        # Projection pour aligner les dimensions RoBERTa -> Config.EMBEDDING_DIM
        self.code_projection = nn.Linear(768, Config.EMBEDDING_DIM)

        # Couche d'attention pour combiner les représentations
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=Config.EMBEDDING_DIM,
            num_heads=Config.N_HEADS,
            dropout=Config.DROPOUT,
            batch_first=True
        )

        # Fusion des représentations AVEC LazyLinear
        self.fusion = nn.Sequential(
            nn.LazyLinear(Config.EMBEDDING_DIM),
            nn.LayerNorm(Config.EMBEDDING_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT)
        )

        # Classifier final AVEC LazyLinear
        self.classifier = nn.Sequential(
            nn.LazyLinear(256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.LazyLinear(128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.LazyLinear(Config.NUM_CLASSES)
        )

    def forward(self, metrics, code_tokens):
        metrics_proj = self.metric_projection(metrics)
        metrics_proj = metrics_proj.unsqueeze(1)
        metrics_encoded = self.metrics_encoder(metrics_proj)

        code_outputs = self.code_decoder(**code_tokens)
        code_embed = code_outputs.last_hidden_state
        code_embed = self.code_projection(code_embed)

        attn_output, _ = self.cross_attention(
            metrics_encoded,
            code_embed,
            code_embed
        )

        combined = torch.cat([metrics_encoded[:, 0, :], attn_output[:, 0, :]], dim=-1)
        fused = self.fusion(combined)

        return self.classifier(fused)

# STRATÉGIE 2: Modèle entièrement pré-entraîné (encoder et decoder pré-entraînés)
class FullPretrainedModel(nn.Module):
    def __init__(self, input_metric_dim=None):
        super().__init__()

        # Encoder PRÉ-ENTRAÎNÉ pour les métriques (adaptation de RoBERTa)
        self.metrics_encoder = RobertaModel.from_pretrained("microsoft/codebert-base")

        # Projection pour les métriques numériques AVEC LazyLinear
        self.metrics_projection = nn.Sequential(
            nn.LazyLinear(Config.EMBEDDING_DIM),
            nn.LayerNorm(Config.EMBEDDING_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT)
        )

        # [2026] Projection des métriques vers la dimension d'entrée de CodeBERT (mémoire, §4.3.2)
        self.metrics_to_codebert = nn.Linear(Config.EMBEDDING_DIM, 768)

        # Decoder PRÉ-ENTRAîNÉ RoBERTa pour le code
        self.code_decoder = RobertaModel.from_pretrained("microsoft/codebert-base")

        # Projection pour aligner les dimensions
        self.code_projection = nn.Linear(768, Config.EMBEDDING_DIM)

        # Couche d'attention croisée pour combiner les deux représentations
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=Config.EMBEDDING_DIM,
            num_heads=Config.N_HEADS,
            dropout=Config.DROPOUT,
            batch_first=True
        )

        # Fusion des représentations AVEC LazyLinear
        self.fusion = nn.Sequential(
            nn.LazyLinear(Config.EMBEDDING_DIM),
            nn.LayerNorm(Config.EMBEDDING_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT)
        )

        # Classifier final AVEC LazyLinear
        self.classifier = nn.Sequential(
            nn.LazyLinear(256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.LazyLinear(128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.LazyLinear(Config.NUM_CLASSES)
        )

    def forward(self, metrics, code_tokens):
        metrics_embedded = self.metrics_projection(metrics).unsqueeze(1)
        metrics_embedded = self.metrics_to_codebert(metrics_embedded)  # [2026] 384 -> 768

        metrics_attention_mask = torch.ones(metrics_embedded.size(0), 1).to(metrics.device)

        metrics_encoder_output = self.metrics_encoder(
            inputs_embeds=metrics_embedded,
            attention_mask=metrics_attention_mask
        )
        metrics_encoded = metrics_encoder_output.last_hidden_state
        metrics_encoded = self.code_projection(metrics_encoded)  # [2026] 768 -> 384, projection partagée

        code_outputs = self.code_decoder(**code_tokens)
        code_encoded = code_outputs.last_hidden_state
        code_encoded = self.code_projection(code_encoded)

        attn_output, _ = self.cross_attention(
            metrics_encoded,
            code_encoded,
            code_encoded
        )

        combined = torch.cat([metrics_encoded[:, 0, :], attn_output[:, 0, :]], dim=-1)
        fused = self.fusion(combined)

        return self.classifier(fused)

# =============================================================================
# NOUVELLES FONCTIONS POUR LA STRATÉGIE MULTI-PROJETS
# =============================================================================

def get_project_versions(project_name, metrics_dir):
    """Retourne toutes les versions disponibles pour un projet, triées par ordre chronologique"""
    all_files = []
    for root, dirs, files in os.walk(metrics_dir):
        for file in files:
            if file.endswith('.csv') and project_name in file:
                # Extraire la version du nom de fichier
                version_match = re.search(r'(\d+\.\d+(?:\.\d+)?)', file)
                if version_match:
                    version = version_match.group(1)
                    all_files.append({
                        'file_path': os.path.join(root, file),
                        'version': version,
                        'project': project_name
                    })

    # Trier par version (ordre chronologique)
    all_files.sort(key=lambda x: [int(part) for part in x['version'].split('.')])
    return all_files

def get_corresponding_code_path(metrics_file_path, source_dir):
    """Trouve le chemin du code source correspondant aux métriques"""
    # Extraire le nom du projet et la version du chemin des métriques
    filename = os.path.basename(metrics_file_path)
    project_match = re.match(r'([a-zA-Z]+)-?(\d+\.\d+(?:\.\d+)?)?', filename)

    if project_match:
        project_name = project_match.group(1)
        version = project_match.group(2) if project_match.group(2) else ""

        # Construire le chemin du code source
        if version:
            code_path = os.path.join(source_dir, project_name, f"{project_name}-{version}")
        else:
            code_path = os.path.join(source_dir, project_name, project_name)

        # Vérifier si le chemin existe
        if os.path.exists(code_path):
            return code_path
        else:
            # Essayer d'autres patterns
            possible_paths = [
                os.path.join(source_dir, project_name),
                os.path.join(source_dir, f"{project_name}-{version}"),
                os.path.join(source_dir, project_name, version),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    return path

    return None

def prepare_cross_project_datasets(target_project, metrics_dir, source_dir, tokenizer):
    """
    Prépare les datasets selon la stratégie multi-projets :
    - Training: (n-1 versions du projet cible) + (toutes les versions des autres projets)
    - Validation: dernière version (n) du projet cible
    """
    print(f"\n  Configuration pour le projet cible: {target_project}")
    print("=" * 60)

    # Obtenir tous les projets disponibles
    all_projects = set()
    for root, dirs, files in os.walk(metrics_dir):
        for file in files:
            if file.endswith('.csv'):
                project_match = re.match(r'([a-zA-Z]+)-?', file)
                if project_match:
                    all_projects.add(project_match.group(1))

    all_projects = sorted(list(all_projects))
    print(f"  Projets disponibles: {all_projects}")

    # Séparer le projet cible des autres projets
    other_projects = [p for p in all_projects if p != target_project]
    print(f"  Projet cible: {target_project}")
    print(f"  Autres projets: {other_projects}")

    # ==================== PRÉPARATION DES DONNÉES D'ENTRAÎNEMENT ====================

    train_features = []
    train_labels = []
    train_filenames = []
    train_code_tokens = {'input_ids': [], 'attention_mask': []}

    # 1. Ajouter les versions n-1 du projet cible
    print(f"\n1. Chargement des versions d'entraînement du projet {target_project}...")
    target_versions = get_project_versions(target_project, metrics_dir)

    if len(target_versions) < 2:
        raise ValueError(f"Le projet {target_project} doit avoir au moins 2 versions pour la validation")

    # Prendre toutes les versions sauf la dernière pour l'entraînement
    train_target_versions = target_versions[:-1]
    val_target_version = target_versions[-1]

    print(f"   - Versions d'entraînement: {[v['version'] for v in train_target_versions]}")
    print(f"   - Version de validation: {val_target_version['version']}")

    for version_info in train_target_versions:
        metrics_path = version_info['file_path']
        code_path = get_corresponding_code_path(metrics_path, source_dir)

        if code_path and os.path.exists(code_path):
            try:
                features, code_tokens, labels, _, filenames = prepare_aligned_dataset(
                    metrics_path, code_path, tokenizer, apply_balancing=False  # Pas de SMOTE individuel
                )

                train_features.append(features)
                train_labels.extend(labels)
                train_filenames.extend(filenames)

                train_code_tokens['input_ids'].append(code_tokens['input_ids'])
                train_code_tokens['attention_mask'].append(code_tokens['attention_mask'])

                print(f"   ✓ Version {version_info['version']}: {len(features)} échantillons")

            except Exception as e:
                print(f"   ✗ Erreur avec {version_info['version']}: {e}")

    # 2. Ajouter TOUTES les versions des autres projets
    print(f"\n2. Chargement des autres projets pour l'entraînement...")
    for other_project in other_projects:
        print(f"     Projet: {other_project}")
        other_versions = get_project_versions(other_project, metrics_dir)

        for version_info in other_versions:
            metrics_path = version_info['file_path']
            code_path = get_corresponding_code_path(metrics_path, source_dir)

            if code_path and os.path.exists(code_path):
                try:
                    features, code_tokens, labels, _, filenames = prepare_aligned_dataset(
                        metrics_path, code_path, tokenizer, apply_balancing=False
                    )

                    train_features.append(features)
                    train_labels.extend(labels)
                    train_filenames.extend(filenames)

                    train_code_tokens['input_ids'].append(code_tokens['input_ids'])
                    train_code_tokens['attention_mask'].append(code_tokens['attention_mask'])

                    print(f"     ✓ {other_project}-{version_info['version']}: {len(features)} échantillons")

                except Exception as e:
                    print(f"     ✗ Erreur avec {other_project}-{version_info['version']}: {e}")

    # ==================== PRÉPARATION DES DONNÉES DE VALIDATION ====================

    print(f"\n3. Chargement des données de validation ({target_project} {val_target_version['version']})...")
    val_metrics_path = val_target_version['file_path']
    val_code_path = get_corresponding_code_path(val_metrics_path, source_dir)

    if not val_code_path or not os.path.exists(val_code_path):
        raise ValueError(f"Chemin du code source non trouvé pour la validation: {val_code_path}")

    val_features, val_code_tokens, val_labels, _, val_filenames = prepare_aligned_dataset(
        val_metrics_path, val_code_path, tokenizer, apply_balancing=False
    )

    # ==================== CONCATÉNATION ET ÉQUILIBRAGE ====================

    # Concaténer les données d'entraînement
    if train_features:
        X_train = np.vstack(train_features)
        y_train = np.array(train_labels)

        # Concaténer les tokens
        all_train_code = {
            'input_ids': torch.cat(train_code_tokens['input_ids']),
            'attention_mask': torch.cat(train_code_tokens['attention_mask'])
        }

        print(f"\n  DONNÉES D'ENTRAÎNEMENT FINALES:")
        print(f"   - Total échantillons: {X_train.shape[0]}")
        print(f"   - Distribution: {Counter(y_train)}")
        print(f"   - Projet cible ({target_project}): {len(train_target_versions)} versions")
        print(f"   - Autres projets: {len(other_projects)} projets")

        # Appliquer SMOTE sur l'ensemble d'entraînement complet
        if len(set(y_train)) > 1:
            print("   - Application de SMOTE sur l'ensemble d'entraînement...")
            X_train, y_train, all_train_code, train_filenames = apply_smote_to_combined_data(
                X_train, y_train, all_train_code, train_filenames
            )
            print(f"   - Après SMOTE: {X_train.shape[0]} échantillons, Distribution: {Counter(y_train)}")

    else:
        raise ValueError("Aucune donnée d'entraînement chargée")

    print(f"\n  DONNÉES DE VALIDATION:")
    print(f"   - Projet: {target_project}")
    print(f"   - Version: {val_target_version['version']}")
    print(f"   - Échantillons: {len(val_features)}")
    print(f"   - Distribution: {Counter(val_labels)}")

    return X_train, all_train_code, y_train, train_filenames, val_features, val_code_tokens, val_labels, val_filenames

def apply_smote_to_combined_data(X, y, code_tokens, filenames):
    """Applique SMOTE sur l'ensemble combiné et réaligne les tokens"""
    print(f"   Distribution avant SMOTE: {Counter(y)}")

    original_size = len(X)

    # Appliquer SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    new_size = len(X_resampled)
    print(f"   Distribution après SMOTE: {Counter(y_resampled)}")

    # Réaligner les tokens et filenames
    if new_size <= original_size:
        code_tokens_resampled = {
            'input_ids': code_tokens['input_ids'][:new_size],
            'attention_mask': code_tokens['attention_mask'][:new_size]
        }
        filenames_resampled = filenames[:new_size]
    else:
        indices = list(range(original_size)) + \
                 list(np.random.randint(0, original_size, new_size - original_size))

        code_tokens_resampled = {
            'input_ids': code_tokens['input_ids'][indices],
            'attention_mask': code_tokens['attention_mask'][indices]
        }
        filenames_resampled = [filenames[i] for i in indices]

    print(f"   ✓ Tokens réalignés: {original_size} → {new_size} échantillons")
    return X_resampled, y_resampled, code_tokens_resampled, filenames_resampled

# =============================================================================
# FONCTIONS EXISTENTES (à garder mais adapter)
# =============================================================================

# [Les fonctions extract_filename_from_path, align_metrics_with_code, load_and_align_data,
#  preprocess_aligned_metrics, prepare_aligned_dataset, calculate_class_weights, evaluate
#  restent identiques à celles du code précédent]

# --- Fonctions reprises de notebooks/05_multimodal/05_hybride_fullpretrained_wpdp.ipynb (cellule 20) ---

# FONCTIONS D'ALIGNEMENT
def extract_filename_from_path(file_path):
    """Extrait le nom de fichier avec plus de flexibilité"""
    if pd.isna(file_path) or file_path is None:
        return []

    base_name = os.path.basename(str(file_path))
    name_without_ext = os.path.splitext(base_name)[0]

    # Nettoyage pour trouver des correspondances
    cleaned_name = re.sub(r'[^a-zA-Z0-9]', '', name_without_ext.lower())

    if not cleaned_name:
        return []

    # Essayer plusieurs formats
    variations = set()
    variations.add(cleaned_name)
    variations.add(cleaned_name.replace('test', '').replace('testcase', ''))
    variations.add(re.sub(r'\d+', '', cleaned_name))

    if '_' in cleaned_name:
        variations.add(cleaned_name.split('_')[0])
    if '.' in cleaned_name:
        variations.add(cleaned_name.split('.')[0])

    variations.add(cleaned_name.replace('_', ''))
    variations.add(re.sub(r'[^a-zA-Z]', '', cleaned_name))

    return [v for v in variations if v and len(v) > 1]

def align_metrics_with_code(df_metrics, code_files_dict):
    """Aligne explicitement les métriques avec le code source correspondant"""
    aligned_metrics = []
    aligned_code = []
    aligned_labels = []
    aligned_filenames = []

    matches_count = 0
    partial_matches_count = 0
    no_matches_count = 0

    # Créer un index des fichiers code pour recherche rapide
    code_index = {}
    for code_filename, code_content in code_files_dict.items():
        code_variations = extract_filename_from_path(code_filename)
        for variation in code_variations:
            if variation:
                code_index[variation] = (code_content, code_filename)

    print(f"✓ Index créé avec {len(code_index)} variations de noms de code")

    for idx, row in df_metrics.iterrows():
        # Extraire le nom de fichier des métriques
        if 'name' in row:
            metric_variations = extract_filename_from_path(str(row['name']))
        else:
            metric_variations = [f"file{idx}"]

        # Chercher le code correspondant
        matching_code = None
        match_type = "none"
        matched_filename = ""

        # Recherche exacte
        for metric_var in metric_variations:
            if metric_var in code_index:
                matching_code, matched_filename = code_index[metric_var]
                match_type = "exact"
                matches_count += 1
                break

        # Recherche partielle
        if matching_code is None:
            for metric_var in metric_variations:
                for code_var, (code_content, filename) in code_index.items():
                    if (metric_var and code_var and
                        (metric_var in code_var or code_var in metric_var)):
                        if len(metric_var) > 3 and len(code_var) > 3:
                            matching_code = code_content
                            matched_filename = filename
                            match_type = "partial"
                            partial_matches_count += 1
                            break
                if matching_code:
                    break

        # Fallback
        if matching_code is None and code_files_dict:
            matching_code, matched_filename = next(iter(code_index.values()))
            match_type = "fallback"
            no_matches_count += 1

        if matching_code is not None:
            # Préparer les métriques
            metric_data = {}
            for col, value in row.items():
                if col not in ['name', 'bug'] and pd.api.types.is_numeric_dtype(type(value)):
                    metric_data[col] = float(value) if pd.notna(value) else 0.0

            aligned_metrics.append(metric_data)
            aligned_code.append(matching_code)
            aligned_labels.append(1 if row.get('bug', 0) > 0 else 0)
            aligned_filenames.append(f"{matched_filename}_{match_type}")

    print(f"✓ Alignement: {matches_count} exacts, {partial_matches_count} partiels, {no_matches_count} fallbacks")
    return aligned_metrics, aligned_code, aligned_labels, aligned_filenames

def load_and_align_data(metrics_path, code_path):
    """Charge et aligne les métriques avec le code source"""
    # Charger les métriques
    df_metrics = pd.read_csv(metrics_path)
    print(f"✓ {len(df_metrics)} métriques chargées depuis {os.path.basename(metrics_path)}")

    # Charger tous les fichiers code
    code_files_dict = {}
    if os.path.exists(code_path):
        for root, dirs, files in os.walk(code_path):
            for file in files:
                if file.endswith('.java'):
                    try:
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            code_content = f.read().strip()
                            if code_content and len(code_content) > 10:
                                code_files_dict[file] = code_content
                    except Exception as e:
                        continue

    print(f"✓ {len(code_files_dict)} fichiers code chargés depuis {os.path.basename(code_path)}")

    if not code_files_dict:
        print("⚠️ Aucun fichier code trouvé!")
        return [], [], [], []

    # Aligner les données
    aligned_metrics, aligned_code, aligned_labels, aligned_filenames = align_metrics_with_code(
        df_metrics, code_files_dict
    )

    print(f"✓ {len(aligned_metrics)} échantillons alignés sur {len(df_metrics)} métriques originales")

    return aligned_metrics, aligned_code, aligned_labels, aligned_filenames

# FONCTIONS DE PRÉTRAITEMENT
def preprocess_aligned_metrics(metrics_list, labels):
    """Prétraite les métriques alignées"""
    if not metrics_list:
        return np.array([]), np.array([]), None

    df_metrics = pd.DataFrame(metrics_list)

    # Gérer les valeurs manquantes
    if df_metrics.isnull().any().any():
        df_metrics = df_metrics.fillna(df_metrics.median())

    # Supprimer les features à variance nulle
    var_threshold = 0.01
    variances = df_metrics.var()
    low_var_cols = variances < var_threshold
    if low_var_cols.any():
        df_metrics = df_metrics.loc[:, ~low_var_cols]
        print(f"✓ {low_var_cols.sum()} colonnes à faible variance supprimées")

    # Normalisation
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_metrics)

    print(f"✓ Métriques prétraitées: {features_scaled.shape}")
    return features_scaled, np.array(labels), scaler

def apply_smote_with_realignment(X, y, code_tokens, filenames):
    """Applique SMOTE et réaligne correctement les tokens de code ET les filenames"""
    print(f"Distribution avant SMOTE: {Counter(y)}")

    original_size = len(X)

    # Appliquer SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    new_size = len(X_resampled)
    print(f"Distribution après SMOTE: {Counter(y_resampled)}")

    # Réaligner les tokens de code ET les filenames
    if new_size <= original_size:
        code_tokens_resampled = {
            'input_ids': code_tokens['input_ids'][:new_size],
            'attention_mask': code_tokens['attention_mask'][:new_size]
        }
        filenames_resampled = filenames[:new_size]
    else:
        indices = list(range(original_size)) + \
                 list(np.random.randint(0, original_size, new_size - original_size))

        code_tokens_resampled = {
            'input_ids': code_tokens['input_ids'][indices],
            'attention_mask': code_tokens['attention_mask'][indices]
        }
        filenames_resampled = [filenames[i] for i in indices]

    print(f"✓ Tokens et filenames réalignés: {original_size} → {new_size} échantillons")
    return X_resampled, y_resampled, code_tokens_resampled, filenames_resampled

def calculate_class_weights(labels):
    """Calcule les poids de classe pour équilibrer la loss"""
    class_counts = Counter(labels)
    total_samples = len(labels)
    n_classes = len(class_counts)

    weights = {}
    for class_id, count in class_counts.items():
        weights[class_id] = total_samples / (n_classes * count)

    weight_tensor = torch.FloatTensor([weights[i] for i in sorted(weights.keys())])
    print(f"✓ Poids des classes: {weights}")
    return weight_tensor

def prepare_aligned_dataset(metrics_path, code_path, tokenizer, apply_balancing=True):
    """Prépare un dataset avec alignement explicite"""
    # Charger et aligner les données
    metrics_list, code_lines, labels, filenames = load_and_align_data(metrics_path, code_path)

    if not metrics_list:
        raise ValueError("Aucune donnée alignée trouvée")

    # Prétraiter les métriques
    features, labels, scaler = preprocess_aligned_metrics(metrics_list, labels)

    # Tokeniser le code AVANT SMOTE
    original_code_tokens = tokenizer(
        code_lines,
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=Config.MAX_CODE_LENGTH
    )

    # Équilibrer les classes SI demandé avec réalignement correct
    if apply_balancing and len(set(labels)) > 1:
        try:
            features, labels, code_tokens, filenames = apply_smote_with_realignment(
                features, labels, original_code_tokens, filenames
            )
            print("✓ SMOTE appliqué avec réalignement correct")
            return features, code_tokens, labels, scaler, filenames

        except Exception as e:
            print(f"Erreur équilibrage: {e}, utilisation des données originales")

    return features, original_code_tokens, labels, scaler, filenames

# FONCTIONS D'ÉVALUATION
def evaluate(model, dataloader, criterion=None):
    """Évaluation robuste du modèle"""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for metrics, code_tokens, labels, _ in dataloader:
            metrics = metrics.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)

            code_tokens_dict = {
                'input_ids': code_tokens['input_ids'].to(Config.DEVICE),
                'attention_mask': code_tokens['attention_mask'].to(Config.DEVICE)
            }

            outputs = model(metrics, code_tokens_dict)

            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculer les métriques
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    try:
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    except:
        f1 = 0.0

    accuracy = accuracy_score(all_labels, all_preds)

    # Calcul robuste de sensibilité et spécificité
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)

        if sens > 0 and spec > 0:
            g_mean = gmean([sens, spec])
        else:
            g_mean = 0.0
    else:
        sens = spec = g_mean = 0.0

    avg_loss = total_loss / len(all_labels) if criterion is not None and len(all_labels) > 0 else None

    return cm, f1, sens, spec, g_mean, accuracy, avg_loss, all_preds, all_labels


# FONCTION D'ENTRAÎNEMENT ADAPTÉE POUR MULTI-PROJETS
def train_model_multi_project(model_type="hybrid", target_project="ant"):
    """Entraîne un modèle avec la stratégie multi-projets"""
    print(f"\n{'='*60}")
    print(f"ENTRAÎNEMENT MULTI-PROJETS: {model_type.upper()}")
    print(f"PROJET CIBLE: {target_project.upper()}")
    print(f"{'='*60}")

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    # Chemins des données
    metrics_dir = Config.DATA_DIR  # [2026] chemins configurables
    source_dir = Config.SOURCE_DIR

    # Préparer les datasets selon la stratégie multi-projets
    X_train, all_train_code, y_train, train_filenames, X_val, val_code_tokens, y_val, val_filenames = prepare_cross_project_datasets(
        target_project, metrics_dir, source_dir, tokenizer
    )

    # Ajustement automatique si mismatch
    min_length = min(len(X_train), len(y_train), len(train_filenames),
all_train_code['input_ids'].shape[0])
    if min_length < len(X_train):
        X_train = X_train[:min_length]
        y_train = y_train[:min_length]
        train_filenames = train_filenames[:min_length]
        all_train_code = {
            'input_ids': all_train_code['input_ids'][:min_length],
            'attention_mask': all_train_code['attention_mask'][:min_length]
        }
        print(f"✓ Données d'entraînement ajustées à {min_length} échantillons")

    min_length_val = min(len(X_val), len(y_val), len(val_filenames),
val_code_tokens['input_ids'].shape[0])
    if min_length_val < len(X_val):
        X_val = X_val[:min_length_val]
        y_val = y_val[:min_length_val]
        val_filenames = val_filenames[:min_length_val]
        val_code_tokens = {
            'input_ids': val_code_tokens['input_ids'][:min_length_val],
            'attention_mask': val_code_tokens['attention_mask'][:min_length_val]
        }
        print(f"✓ Données de validation ajustées à {min_length_val} échantillons")

    # Créer les datasets
    train_ds = AlignedBugDataset(X_train, all_train_code, y_train, train_filenames)
    val_ds = AlignedBugDataset(X_val, val_code_tokens, y_val, val_filenames)

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE)

    # Initialiser le modèle
    if model_type == "full_pretrained":
        model = FullPretrainedModel().to(Config.DEVICE)
        model_name = "Full Pretrained"
    else:
        model = HybridTransBugModel().to(Config.DEVICE)
        model_name = "Hybrid"

    print(f"✓ Modèle {model_name} initialisé")

    # Initialiser les LazyLinear
    print("  Initialisation des couches LazyLinear...")
    with torch.no_grad():
        sample_metrics = torch.randn(2, X_train.shape[1]).to(Config.DEVICE)
        sample_code_tokens = {
            'input_ids': torch.randint(0, 1000, (2, 10)).to(Config.DEVICE),
            'attention_mask': torch.ones(2, 10).to(Config.DEVICE)
        }
        _ = model(sample_metrics, sample_code_tokens)
    print("✓ LazyLinear initialisés")

    # [Le reste de l'entraînement reste identique...]

    # Optimiseur avec learning rates différenciés
    if model_type == "full_pretrained":
        optimizer = torch.optim.AdamW([
            {'params': model.metrics_encoder.parameters(), 'lr': Config.LEARNING_RATE * 0.1},
            {'params': model.code_decoder.parameters(), 'lr': Config.LEARNING_RATE * 0.1},
            {'params': [p for n, p in model.named_parameters()
                       if 'metrics_encoder' not in n and 'code_decoder' not in n],
             'lr': Config.LEARNING_RATE}
        ], weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW([
            {'params': model.code_decoder.parameters(), 'lr': Config.LEARNING_RATE * 0.1},
            {'params': [p for n, p in model.named_parameters() if 'code_decoder' not in n],
             'lr': Config.LEARNING_RATE}
        ], weight_decay=0.01)

    # Scheduler et criterion
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    class_weights = calculate_class_weights(y_train).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights,
label_smoothing=Config.LABEL_SMOOTHING)

    # [Le reste de l'entraînement identique...]

    # Listes pour le monitoring
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    f1_scores, gmean_scores = [], []

    best_f1 = 0
    patience_counter = 0
    best_model_state = None

    # ENTRAÎNEMENT
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0
        train_preds, train_labels_list = [], []

        for batch_idx, (metrics, code_tokens, labels, _) in enumerate(train_loader):
            metrics = metrics.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)

            code_tokens_dict = {
                'input_ids': code_tokens['input_ids'].to(Config.DEVICE),
                'attention_mask': code_tokens['attention_mask'].to(Config.DEVICE)
            }

            outputs = model(metrics, code_tokens_dict)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels_list.extend(labels.cpu().numpy())

        # Métriques d'entraînement
        train_loss /= len(train_preds) if train_preds else 1
        train_acc = accuracy_score(train_labels_list, train_preds) if train_labels_list else 0
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Évaluation
        cm, f1, sens, spec, gmean, val_acc, val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion)
        val_losses.append(val_loss if val_loss else 0)
        val_accuracies.append(val_acc)
        f1_scores.append(f1)
        gmean_scores.append(gmean)

        scheduler.step(f1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{Config.EPOCHS}:")
            print(f"  Train: Loss={train_loss:.4f}, Accuracy={train_acc:.4f}")
            print(f"  Val: Loss={val_loss:.4f}, Accuracy={val_acc:.4f}, F1={f1:.4f}, G-Mean={gmean:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            torch.save(model.state_dict(),
f"{Config.SAVE_PATH}best_model_{model_type}_{target_project}.pt")
            print(f"  *** Nouveau meilleur F1: {f1:.4f} ***")
        else:
            patience_counter += 1

        if patience_counter >= 10:
            print(f"Early stopping déclenché à l'epoch {epoch+1}")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)

    print(f"\n{model_name.upper()} - Meilleur F1 score: {best_f1:.4f}")

    # Sauvegarder les métriques
    results = {
        'model_type': model_type,
        'model_name': model_name,
        'target_project': target_project,
        'best_f1': best_f1,
        'final_accuracy': val_accuracies[-1] if val_accuracies else 0,
        'final_sensitivity': sens,
        'final_specificity': spec,
        'final_gmean': gmean,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'f1_scores': f1_scores,
        'gmean_scores': gmean_scores
    }

    joblib.dump(results,
f"{Config.SAVE_PATH}training_results_{model_type}_{target_project}.pkl")

    return model, results

# FONCTION PRINCIPALE MULTI-PROJETS
def main_multi_project():
    """Fonction principale pour la validation multi-projets"""
    print("  VALIDATION MULTI-PROJETS - STRATÉGIE LEAVE-LAST-VERSION-OUT")
    print("=" * 80)
    print("STRATÉGIE:")
    print("• Pour chaque projet cible:")
    print("  - Training: (n-1 versions du projet cible) + (toutes les versions des autres projets)")
    print("  - Validation: dernière version (n) du projet cible")
    print("=" * 80)

    # Liste des projets à évaluer
    target_projects = ["ant", "camel", "jedit", "log4j", "lucene", "poi", "synapse", "velocity", "xalan", "xerces"]  # [2026] synapse ajouté

    all_results = {}

    for target_project in target_projects:
        try:
            print(f"\n{'#'*80}")
            print(f"ÉVALUATION DU PROJET: {target_project.upper()}")
            print(f"{'#'*80}")

            # Entraîner le modèle hybride
            print(f"\n1. STRATÉGIE HYBRIDE - {target_project}")
            model_hybrid, results_hybrid = train_model_multi_project("hybrid", target_project)

            # Entraîner le modèle full pretrained
            print(f"\n2. STRATÉGIE FULL PRETRAINED - {target_project}")
            model_full, results_full = train_model_multi_project("full_pretrained", target_project)

            # Stocker les résultats
            all_results[target_project] = {
                'hybrid': results_hybrid,
                'full_pretrained': results_full
            }

            # Comparaison pour ce projet
            print(f"\n{'='*60}")
            print(f"COMPARAISON POUR {target_project.upper()}")
            print(f"{'='*60}")

            hybrid_f1 = results_hybrid['best_f1']
            full_f1 = results_full['best_f1']
            best_model = "HYBRIDE" if hybrid_f1 > full_f1 else "FULL PRETRAINED"
            advantage = abs(hybrid_f1 - full_f1)

            print(f"Hybride F1: {hybrid_f1:.4f}")
            print(f"Full Pretrained F1: {full_f1:.4f}")
            print(f"  Meilleur: {best_model} (+{advantage:.4f})")

        except Exception as e:
            print(f"  Erreur avec le projet {target_project}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # RAPPORT FINAL
    print(f"\n{'#'*80}")
    print("RAPPORT FINAL - COMPARAISON MULTI-PROJETS")
    print(f"{'#'*80}")

    print(f"{'Projet':<15} {'Hybride F1':<12} {'Full Pretrained F1':<18} {'Meilleur':<15} {'Avantage':<10}")
    print("-" * 80)

    hybrid_wins = 0
    full_wins = 0

    for project in target_projects:
        if project in all_results:
            hybrid_f1 = all_results[project]['hybrid']['best_f1']
            full_f1 = all_results[project]['full_pretrained']['best_f1']

            if hybrid_f1 > full_f1:
                best_model = "Hybride"
                advantage = hybrid_f1 - full_f1
                hybrid_wins += 1
            else:
                best_model = "Full Pretrained"
                advantage = full_f1 - hybrid_f1
                full_wins += 1

            print(f"{project:<15} {hybrid_f1:<12.4f} {full_f1:<18.4f} {best_model:<15} +{advantage:.4f}")

    print("-" * 80)
    print(f"RÉSUMÉ: Hybride {hybrid_wins} - {full_wins} Full Pretrained")

    if hybrid_wins > full_wins:
        print("  CONCLUSION: La stratégie Hybride est globalement meilleure")
    elif full_wins > hybrid_wins:
        print("  CONCLUSION: La stratégie Full Pretrained est globalement meilleure")
    else:
        print("  CONCLUSION: Les deux stratégies sont équivalentes")

    # Sauvegarder le rapport complet
    joblib.dump(all_results, f"{Config.SAVE_PATH}multi_project_comparison.pkl")
    print(f"\n  Rapport complet sauvegardé: {Config.SAVE_PATH}multi_project_comparison.pkl")

if __name__ == "__main__":
    main_multi_project()