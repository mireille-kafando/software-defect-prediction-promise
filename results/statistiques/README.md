# Analyses statistiques

Cette page rassemble les résultats des tests statistiques exécutés dans les notebooks d'analyse du mémoire. Les valeurs sont recopiées des sorties d'exécution, sans recalcul :

- CNN1D : [`02_analyse_cnn1d/analyse_statistique_cnn1d`](../../notebooks/02_analyse_cnn1d/analyse_statistique_cnn1d.ipynb) ;
- Transformer encodeur : [`04_analyse_transformer/analyse_statistique_transformer`](../../notebooks/04_analyse_transformer/analyse_statistique_transformer.ipynb).

Dans le dépôt, ces notebooks sont présentés sans leurs sorties pour rester légers. Les cellules citées ci-dessous permettent de retrouver chaque test.

*English summary at the end.*

## Démarche

Chaque comparaison suit la même logique :

1. **Normalité** de chaque série (ou des différences appariées) par le test de **Shapiro-Wilk**, avec un graphique Q-Q.
2. **Test de comparaison** choisi selon le résultat :
   - test paramétrique si la normalité est acceptée : t de Student, ANOVA à un facteur ou ANOVA à mesures répétées ;
   - test non paramétrique sinon : Wilcoxon, Mann-Whitney, Kruskal-Wallis ou Friedman.
3. **Comparaisons par paires** (Wilcoxon, Tukey HSD) pour situer les écarts entre configurations.

Toutes les comparaisons portent sur la **G-Mean** obtenue sur les 11 projets PROMISE. Le seuil est α = 0,05.

## 1. Corpus : distribution des métriques

Métriques des 41 versions, sur les 14 866 classes chargées dans le notebook (cellule 80) :

| Métrique | Shapiro-Wilk W | p | Asymétrie | Aplatissement |
|---|---|---|---|---|
| LOC | 0,329 | < 10⁻¹⁰⁰ | 12,3 | 272,0 |
| NOC | 0,132 | < 10⁻¹⁰⁰ | 19,5 | 578,4 |
| CBO | 0,442 | < 10⁻¹⁰⁰ | 8,6 | 134,3 |
| MAX_CC | 0,373 | < 10⁻¹⁰⁰ | 11,7 | 229,2 |

Corrélations de Spearman : LOC–MAX_CC ρ = 0,73 ; CBO–LOC ρ = 0,44 ; CBO–MAX_CC ρ = 0,39 ; CBO–NOC ρ = 0,24 ; NOC–LOC ρ = 0,09 ; NOC–MAX_CC ρ = 0,09. Toutes ont p < 0,001.

Les métriques sont très asymétriques, avec des queues lourdes. C'est ce qui justifie la standardisation et les tests non paramétriques sur les données brutes.

<p align="center"><img src="figures/corpus_qqplot_metriques.png" width="95%" alt="Graphiques Q-Q de LOC, NOC, CBO et MAX_CC"></p>
<p align="center">
  <img src="figures/corpus_correlation_metriques.png" width="40%" alt="Matrice de corrélation de Spearman">
  <img src="figures/corpus_evolution_loc_par_projet.png" width="55%" alt="Évolution de la taille moyenne des classes (LOC) par projet et par version">
</p>

## 2. CNN1D

### 2.1 Effet de la standardisation (SG, SV, SR)

| Architecture | Protocole | Normalité (p Shapiro SG / SV / SR) | Test | Statistique | p |
|---|---|---|---|---|---|
| Simple | WPDP | 0,40 / 0,26 / 0,25 | ANOVA à un facteur | F = 0,143 | 0,868 |
| Dense | WPDP | 0,93 / **0,027** / 0,26 | Kruskal-Wallis | H = 0,391 | 0,823 |
| Simple | CPDP | 0,51 / 0,24 / 0,74 | ANOVA à un facteur | F = 0,520 | 0,600 |
| Dense | CPDP | 0,08 / **0,010** / 0,18 | Kruskal-Wallis | H = 0,181 | 0,914 |
| Dense | CPDP | — | Friedman | χ² = 0,545 | 0,761 |

Cellules 8, 10, 12, 14 et 26. **Aucun effet significatif de la standardisation sur la G-Mean.**

Sur l'exactitude moyenne, en revanche, la standardisation a un effet significatif en CPDP :
- CNN1D simple : Kruskal-Wallis H = 13,90, p = 0,001 ;
- CNN1D dense : H = 10,54, p = 0,005 ;
- CNN1D dense, test de Friedman : χ² = 6,73, p = 0,035.

### 2.2 WPDP contre CPDP, à standardisation égale (tests appariés)

| Architecture | Standardisation | G-Mean : test, p | Exactitude : test, p |
|---|---|---|---|
| Simple | SG | t apparié, p = 0,236 | t apparié, **p = 0,001** |
| Simple | SV | Wilcoxon, p = 0,898 | t apparié, **p = 0,047** |
| Simple | SR | t apparié, p = 0,470 | t apparié, p = 0,095 |
| Dense | SG | Wilcoxon, p = 0,898 | t apparié, **p < 0,001** |
| Dense | SV | t apparié, p = 0,942 | t apparié, **p = 0,013** |
| Dense | SR | t apparié, p = 0,741 | t apparié, **p = 0,023** |

Cellules 16 et 29. **La G-Mean ne diffère jamais entre WPDP et CPDP.** L'exactitude, elle, diffère le plus souvent : c'est la raison pour laquelle le mémoire retient la G-Mean, moins sensible au déséquilibre des classes.

<p align="center"><img src="figures/cnn1d_simple_gmean_wpdp_vs_cpdp.png" width="95%" alt="G-Mean du CNN1D simple, WPDP contre CPDP, pour les trois standardisations"></p>
<p align="center"><img src="figures/cnn1d_simple_exactitude_wpdp_vs_cpdp.png" width="95%" alt="Exactitude du CNN1D simple, WPDP contre CPDP, pour les trois standardisations"></p>
<p align="center"><img src="figures/cnn1d_dense_gmean_wpdp_vs_cpdp.png" width="85%" alt="G-Mean du CNN1D dense, WPDP contre CPDP, pour les trois standardisations"></p>

Ordre des boîtes à moustaches : cas 1 = SG, cas 2 = SV, cas 3 = SR.

### 2.3 Simple contre dense, et WPDP contre CPDP en moyenne

Ces comparaisons partent de fichiers Excel de moyennes par projet (cellules 2 à 6) :

| Comparaison | Normalité (p) | Test | t | p |
|---|---|---|---|---|
| Simple contre dense, WPDP | 0,93 / 0,13 | t de Student | 0,359 | 0,723 |
| Simple contre dense, CPDP | 0,43 / 0,50 | t de Student | 0,909 | 0,374 |
| WPDP contre CPDP, CNN1D dense | 0,13 / 0,50 | t de Student | 0,127 | 0,900 |
| WPDP contre CPDP, CNN1D simple | 0,93 / 0,43 | t de Student | −0,339 | 0,738 |

<p align="center"><img src="figures/cnn1d_qqplot_simple_vs_dense_wpdp.png" width="60%" alt="Graphiques Q-Q, CNN1D simple et dense"></p>

## 3. Transformer encodeur

### 3.1 Protocole × technique d'équilibrage (6 configurations, cellule 25)

| Configuration | G-Mean moyenne (%) | Shapiro-Wilk p |
|---|---|---|
| WPDP + SMOTE | 62,23 | 0,238 |
| WPDP + sous-échantillonnage | 63,75 | 0,422 |
| WPDP + suppression des valeurs aberrantes | 61,64 | 0,849 |
| CPDP + SMOTE | 61,76 | 0,340 |
| CPDP + sous-échantillonnage | 63,07 | 0,175 |
| CPDP + suppression des valeurs aberrantes | 62,88 | 0,256 |

- **ANOVA à mesures répétées** : p = 0,967.
- **Friedman** : χ² = 1,338, p = 0,931.
- **Wilcoxon par paires** (9 paires) : aucune paire significative ; la p la plus faible est 0,102 (CPDP SMOTE contre CPDP sous-échantillonnage).

<p align="center">
  <img src="figures/transformer_gmean_six_configurations.png" width="48%" alt="Distribution de la G-Mean par configuration">
  <img src="figures/transformer_gmean_moyennes_ic95.png" width="48%" alt="G-Mean moyenne et intervalles de confiance à 95 %">
</p>
<p align="center"><img src="figures/transformer_profil_par_projet.png" width="70%" alt="Profil de la G-Mean de chaque projet selon la configuration"></p>

### 3.2 Autres comparaisons

| Comparaison | ANOVA MR, p | Friedman | Cellule |
|---|---|---|---|
| Techniques d'équilibrage en CPDP (3 techniques) | 0,300 | χ² = 2,279, p = 0,320 | 18 |
| Dimension du modèle 20 contre 40, WPDP et CPDP (4 configurations) | 0,894 | χ² = 0,600, p = 0,896 | 28 |

<p align="center"><img src="figures/transformer_dimension_20_vs_40.png" width="55%" alt="G-Mean selon la dimension du modèle"></p>

## 4. Conclusion des tests

Aucune des comparaisons de G-Mean n'est significative au seuil de 5 % : standardisation, technique d'équilibrage, protocole WPDP/CPDP, profondeur du CNN1D, dimension du Transformer. Les écarts entre configurations (1 à 2 points) restent faibles devant la variabilité entre projets (écart type d'environ 7 points). Ce constat rejoint la conclusion du mémoire : les caractéristiques des projets pèsent davantage que les choix méthodologiques.

## Ce qui ne figure pas dans les notebooks conservés

Le mémoire présente aussi plusieurs analyses absentes des sorties des notebooks retrouvés :
- le **W de Kendall** ;
- les tests post-hoc de **Nemenyi** ;
- la **décomposition de variance par ANOVA à trois facteurs** (projet × standardisation × architecture, tableaux 16 et 17) ;
- le **test de Friedman entre les cinq architectures**.

Elles ont été calculées en dehors de ces notebooks. Les G-Mean nécessaires pour les refaire se trouvent dans les fichiers CSV de [`results/`](../).

## Écarts relevés dans les analyses

Les notebooks sont présentés tels qu'ils ont été exécutés. Une relecture a mis en évidence les points suivants :

- **Tests pour échantillons indépendants sur des données appariées.** Les 11 projets sont les mêmes d'une configuration à l'autre. Pourtant :
  - les cellules 2 à 6 utilisent le t de Student pour échantillons indépendants (`ttest_ind`) ;
  - les cellules 8 à 14 utilisent l'ANOVA à un facteur et Kruskal-Wallis, qui supposent des groupes indépendants.

  Les tests appariés adaptés à ce plan (t apparié, Wilcoxon, ANOVA à mesures répétées, Friedman) sont ceux des cellules 16, 26 et 29 et du notebook du Transformer. Pour la G-Mean, ils mènent à la même conclusion.
- **Comparaisons multiples.** Les nombreuses comparaisons par paires (Wilcoxon) sont faites sans correction de Bonferroni ni de Holm. Comme aucune n'est significative pour la G-Mean, la conclusion n'en dépend pas. En revanche, les écarts d'exactitude significatifs de la section 2.2 seraient en partie à réexaminer avec une correction.
- **Effectif du corpus.** L'analyse des métriques porte sur 14 866 classes, alors que le corpus en compte 15 775. Une partie des lignes n'a pas été chargée ou a été écartée dans cette cellule.

## English summary

Results of the statistical tests run in the thesis analysis notebooks, copied from their execution outputs:

- **Method:** Shapiro-Wilk normality checks, then a parametric test (t-test, ANOVA, repeated-measures ANOVA) or a non-parametric one (Wilcoxon, Mann-Whitney, Kruskal-Wallis, Friedman).
- **Findings:** none of the G-Mean comparisons is significant at α = 0.05: standardization, balancing technique, WPDP vs CPDP, CNN1D depth, Transformer width. For the Transformer encoder, Friedman χ² = 1.34 (p = 0.93) across six configurations. Accuracy, by contrast, often differs between WPDP and CPDP, which motivates the use of G-Mean. Code metrics are heavily skewed and non-normal (Shapiro-Wilk p < 10⁻¹⁰⁰).
- **Known issues:** some cells apply independent-sample tests to paired data, and pairwise comparisons are not corrected for multiplicity.
