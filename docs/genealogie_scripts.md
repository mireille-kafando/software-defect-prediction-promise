# Généalogie des scripts multimodaux

Les scripts PyTorch des modèles multimodaux (métriques + code source) ont évolué par versions successives. Leur historique complet tient dans un document de travail de 288 pages (« import torch »), où l'autrice a collé chaque génération de code. Cette page relie les familles de ce document aux fichiers publiés et documente les écarts entre le code et le mémoire.

## Correspondance familles → fichiers du dépôt

| Famille du document de travail | Logique | Fichier du dépôt | Statut |
|---|---|---|---|
| **A** — Fusion multimodale de base (`MultiModalBugPredictor`) | Encodeur dense des métriques + CodeBERT, fusion par un Transformer sur deux jetons, WPDP | Non publié | Prototype, non retenu |
| **B** — Audit de l'alignement et du prétraitement (`MultimodalDataValidator`, `MultimodalPreprocessor`) | Qualité d'appariement métriques ↔ code, tokenisation synchronisée | `notebooks/06_figures_memoire/figures_annexes.ipynb`, cellule 9 | Outils de diagnostic |
| **C1** — Alignement explicite par attention (`CrossAttentionAlignedModel`, `HierarchicalAlignedModel`) | Attention croisée et hiérarchique, BorderlineSMOTE | `notebooks/05_multimodal/05_hybride_fullpretrained_wpdp.ipynb`, cellule 13 | Exploratoire |
| **C2–C4** — Correctif SMOTE avec réassociation du code | Propagation des noms de fichiers, SMOTE puis réassociation des jetons | même notebook, cellules 15 et 16 | Exploratoire |
| **D1** — Hybride vs Full Pretrained, version initiale | Dimension métrique explicite (`nn.Linear(input_metric_dim, 384)`) | même notebook, cellule 18 | Version antérieure |
| **D2** — Hybride vs Full Pretrained, révision | `nn.LazyLinear(384)`, CodeBERT pré-entraîné, `Linear(768, 384)`, attention croisée, taux d'apprentissage × 0,1 pour les composants pré-entraînés | même notebook, cellule 20 | **Retenue (WPDP)** |
| **E** — Validation multi-projets | Historique du projet cible + toutes les versions des autres projets, test sur la dernière version cible | `scripts/multimodal_cpdp_hybride_fullpretrained.py` | **Retenue (CPDP)** |
| — Essais CPDP antérieurs (`HybridBugPredictorV1/V2`, 768 dimensions) | Même protocole, autre architecture | `notebooks/05_multimodal/06_pipeline_cpdp_exploratoire.ipynb` | Exploratoire |

Le script de la famille E est une transcription fidèle du document de travail. Seules les coupures de ligne dues à la mise en page ont été recollées. Les fonctions communes que l'autrice indique « identiques à celles du code précédent » sont reprises telles quelles de la cellule 20 du notebook WPDP, où elles figurent à l'identique (vérification ligne à ligne).

## Preuves de rattachement des résultats CPDP

Les G-Mean CPDP du tableau 12 proviennent des fichiers `training_results_<modèle>_<projet>.pkl`, agrégés dans `all_projects_results.pkl`. Le script E est celui qui les produit :

- les clés enregistrées sont exactement celles du dictionnaire `results` du script (`best_f1`, `final_sensitivity`, `final_specificity`, `final_gmean`, `train_losses`, `f1_scores`, `gmean_scores`…) ;
- le nombre d'époques enregistré va de 11 à 23 sur 25. C'est ce qu'impose l'arrêt précoce du script (patience de 10) : 11 époques est le minimum possible, atteint quand le meilleur F1 (non nul) survient dès la première époque ;
- les moyennes recalculées depuis ces fichiers (Hybride 50,89 %, Full Pretrained 54,69 %) sont celles du README.

## Écarts entre le script et les résultats conservés

1. **Full Pretrained ne s'exécute pas tel qu'écrit.** La projection des métriques produit 384 dimensions, mais CodeBERT attend 768 dimensions en entrée (`inputs_embeds`). Le test de forme (CodeBERT à l'architecture identique, poids aléatoires) donne `RuntimeError: The size of tensor a (384) must match the size of tensor b (768)`. Le modèle Hybride, lui, s'exécute. La version qui a produit les résultats Full Pretrained comportait donc une adaptation 384 ↔ 768, comme le décrit le §4.3.2 du mémoire (`nn.Linear(384, 768)`), mais cette version n'a pas été conservée.
2. **Synapse.** La liste `target_projects` du script ne contient pas Synapse, alors que les résultats conservés l'incluent. L'ordre des clés enregistrées diffère aussi légèrement. La version exécutée était donc une variante proche du script publié.

## Points de méthode

Ces points ne modifient pas les tableaux publiés ; ils en précisent les conditions d'interprétation.

1. **Standardisation des modèles multimodaux.** Chaque fichier de version passe par son propre `StandardScaler`, y compris la version de test. Les modèles multimodaux utilisent donc en pratique une standardisation par version (SV) appliquée de façon transductive, et non la standardisation globale (SG, notée « S1 » au tableau 12) indiquée au §2.2.2 et au tableau 12 du mémoire.
2. **Équilibrage.** En WPDP, SMOTE est appliqué séparément à chaque version d'entraînement avant concaténation. En CPDP, il est appliqué une fois à l'ensemble d'entraînement concaténé. SMOTE ne crée pas de code : les exemples synthétiques reçoivent les jetons d'un fichier existant tiré au hasard (`np.random.randint`, sans graine fixée dans le script).
3. **Arrêt précoce sur l'ensemble évalué.** Il n'y a pas de jeu de validation distinct de la dernière version. L'arrêt précoce est piloté par le F1 de cette version. Dans les scripts PyTorch, la copie `state_dict().copy()` est superficielle : la G-Mean rapportée est celle de la dernière époque. Le CNN1D (Keras) utilise `EarlyStopping(monitor='val_loss', restore_best_weights=True)` sur ce même ensemble. L'encodeur Transformer unimodal, entraîné 100 époques sans arrêt précoce, n'est pas concerné.
4. **Appariement de repli.** Lorsqu'aucun fichier Java ne correspond au nom de la classe, la fonction d'alignement associe le premier fichier de code disponible (`match_type = "fallback"`). Le nombre de replis est affiché pendant l'exécution, mais les journaux correspondants n'ont pas été conservés. Une mesure faite après coup sur le corpus réel (voir le README, « Écarts relevés ») indique que la fonction ne relie que 0,7 % des classes à leur propre fichier.
5. **Troncature.** Les ensembles sont tronqués à la plus petite longueur commune en cas de décalage entre métriques, jetons et noms de fichiers.
6. **Paramètres divergents entre versions.** Époques (15 à 25), taux d'apprentissage (1e-7 à 2e-5), détecteur d'anomalies et taux de contamination varient d'une génération à l'autre. Ce sont des choix expérimentaux distincts, pas des doublons.

