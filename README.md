# Prédiction des défauts logiciels par apprentissage profond

**CNN1D, Transformer et modèles hybrides métriques + CodeBERT sur les projets PROMISE**

Code et résultats de mon mémoire de maîtrise en mathématiques et informatique appliquées (concentration génie logiciel), Université du Québec à Trois-Rivières (UQTR), 2026.

[English summary below](#english-summary)

<!-- À COMPLÉTER : lien vers le mémoire (dépôt institutionnel UQTR) une fois disponible -->

---

## En bref

Prédire quelles classes d'un logiciel contiendront des défauts aide les équipes à cibler leurs tests et leurs revues de code. Le problème est difficile : les classes défectueuses sont minoritaires, si bien qu'un modèle peut afficher une bonne exactitude tout en manquant la plupart des défauts.

Ce projet compare trois familles de modèles d'apprentissage profond sur les métriques de code de 11 projets Java open source :

| Étape | Modèle | Entrées | Outils |
|---|---|---|---|
| 1 | **CNN1D** | Métriques logicielles | TensorFlow / Keras |
| 2 | **Transformer** (encodeur) | Métriques logicielles | PyTorch |
| 3 | **Modèles hybrides** : *Hybrid* (encodeur entraîné de zéro) et *Full Pretrained* (encodeur pré-entraîné) | Métriques + code source encodé par **CodeBERT** | PyTorch, Hugging Face Transformers |

Les résultats des modèles CNN1D et Transformer font l'objet d'une analyse statistique (Shapiro-Wilk, Friedman, Wilcoxon, Nemenyi).

**Points clés**

- Le déséquilibre des classes est traité par rééchantillonnage (SMOTE, Borderline-SMOTE, ADASYN, SMOTE-Tomek) et par la détection de valeurs aberrantes (IsolationForest).
- La métrique principale est la **G-Mean** (moyenne géométrique de la sensibilité et de la spécificité), plus fiable que l'exactitude sur des données déséquilibrées.
- Deux protocoles d'évaluation : **intra-version** (entraînement et test sur une même version) et **inter-version** (entraînement sur les versions antérieures d'un projet, test sur une version ultérieure). Les modèles hybrides sont évalués en inter-version.

## Résultats

### CNN1D : G-Mean et exactitude moyennes sur 11 projets

Trois configurations expérimentales (cas 1 à 3, détaillées dans le mémoire), évaluées en intra-version et en inter-version.

<!-- À COMPLÉTER : nommer les cas 1, 2 et 3 -->

| Évaluation | Indicateur | Cas 1 | Cas 2 | Cas 3 |
|---|---|---|---|---|
| Intra-version | G-Mean | 0,637 | 0,634 | 0,626 |
| Intra-version | Exactitude | 0,806 | 0,816 | 0,822 |
| Inter-version | G-Mean | 0,619 | 0,631 | 0,636 |
| Inter-version | Exactitude | 0,725 | 0,764 | 0,766 |

L'écart entre l'exactitude (0,73 à 0,82) et la G-Mean (0,62 à 0,64) illustre l'effet du déséquilibre des classes. Le détail par projet se trouve dans [`results/cnn1d_intraversion.csv`](results/cnn1d_intraversion.csv) et [`results/cnn1d_interversion.csv`](results/cnn1d_interversion.csv).

### Modèles hybrides métriques + CodeBERT : G-Mean par projet

| Projet | Hybrid | Full Pretrained |
|---|---|---|
| ant | 0,700 | **0,728** |
| camel | **0,353** | 0,251 |
| jedit | 0,638 | **0,647** |
| log4j | 0,462 | **0,477** |
| lucene | 0,612 | **0,626** |
| poi | 0,691 | **0,753** |
| synapse | 0,456 | **0,638** |
| velocity | 0,282 | **0,548** |
| xalan | **0,579** | 0,537 |
| xerces | **0,316** | 0,265 |
| **Moyenne** | 0,509 | **0,547** |

- Le modèle *Full Pretrained* obtient la meilleure G-Mean sur 7 projets sur 10.
- Les performances varient fortement d'un projet à l'autre. Sur camel et xerces, la sensibilité reste très faible (moins de 0,13) : le modèle détecte peu de classes défectueuses malgré le rééchantillonnage.
- Exactitude, sensibilité et spécificité pour chaque projet : [`results/codebert_final_results.csv`](results/codebert_final_results.csv).

Ces deux séries d'expériences ne sont pas directement comparables : elles n'utilisent ni les mêmes projets (ivy n'apparaît que dans la première) ni exactement le même protocole.

### Exemple : courbes d'apprentissage et métriques finales (projet ant)

![Comparaison Hybrid vs Full Pretrained sur le projet ant](results/figures/comparison_ant.png)

Les figures des 10 projets se trouvent dans [`results/figures/`](results/figures/).

## Organisation du dépôt

```
notebooks/
├── 01_cnn1d/                 CNN1D sur les métriques (exploration, couche dense, version finale)
├── 02_analyse_cnn1d/         Analyse statistique des résultats CNN1D
├── 03_transformer/           Encodeur Transformer sur les métriques
├── 04_analyse_transformer/   Analyse statistique des résultats Transformer
├── 05_hybride_codebert/      Fusion métriques + CodeBERT, rééchantillonnage, alignement métriques/code
├── 06_pipeline_final/        Expériences finales sur les 10 projets (Hybrid vs Full Pretrained)
└── 07_figures_memoire/       Figures, tableaux et statistiques descriptives du mémoire
results/
├── cnn1d_intraversion.csv
├── cnn1d_interversion.csv
├── codebert_final_results.csv
└── figures/                  Une figure de comparaison par projet
data/README.md                Obtenir les données PROMISE
```

Chaque notebook commence par une cellule qui décrit son rôle dans le mémoire.

## Reproduire les expériences

1. Obtenir les données PROMISE et le code source des projets (voir [`data/README.md`](data/README.md)).
2. Ouvrir un notebook dans Google Colab (GPU recommandé pour les modèles CodeBERT), ou localement après `pip install -r requirements.txt`.
3. Adapter les chemins `/content/drive/MyDrive/bug/...` à l'emplacement des données.

Ces notebooks sont des notebooks de recherche : ils gardent la trace des itérations successives et ne constituent pas une bibliothèque prête à l'emploi.

## Compétences mises en œuvre

Python · TensorFlow/Keras · PyTorch · Hugging Face Transformers (CodeBERT) · scikit-learn · imbalanced-learn · pandas · SciPy/statsmodels (tests statistiques) · Matplotlib/Seaborn · Google Colab

## Auteure

**Mireille Kafando**, M. Sc. en mathématiques et informatique appliquées (génie logiciel), UQTR

---

## English summary

Code and results from my master's thesis (Université du Québec à Trois-Rivières, 2026) on **software defect prediction** with deep learning on the PROMISE dataset (11 open-source Java projects).

- **Models:** 1D CNN (Keras) and Transformer encoder (PyTorch) on software metrics, then hybrid models that fuse software metrics with source code embeddings from **CodeBERT**. The *Hybrid* variant uses an encoder trained from scratch, the *Full Pretrained* variant a pretrained encoder.
- **Class imbalance:** SMOTE, Borderline-SMOTE, ADASYN, SMOTE-Tomek and IsolationForest outlier removal. Models are evaluated with **G-Mean** and validated across project versions.
- **Results:** the 1D CNN reaches an average G-Mean of about 0.62–0.64 across 11 projects. Among the CodeBERT-based models, *Full Pretrained* achieves the best G-Mean on 7 of 10 projects (mean 0.547 vs 0.509). Results vary widely between projects.
- **Statistical analysis:** Shapiro-Wilk, Friedman, Wilcoxon and Nemenyi tests.

See the tables above and the `results/` folder for details. Data are not redistributed; see `data/README.md`.
