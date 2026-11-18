# Données

Les données ne sont pas incluses dans ce dépôt : elles appartiennent à leurs auteurs.

## Métriques PROMISE

Les fichiers de métriques (une ligne par classe Java, avec le nombre de défauts) proviennent du jeu de données PROMISE pour les projets ant, camel, ivy, jedit, log4j, lucene, poi, synapse, velocity, xalan et xerces.

Les notebooks attendent la structure suivante :

```
bug/bug-data/
├── ant/
│   ├── ant-1.3.csv
│   ├── ...
│   └── ant-1.7.csv
├── camel/
└── ...
```

## Code source (modèles CodeBERT)

Les modèles hybrides ont besoin du code source Java de chaque version, disponible dans les archives officielles des projets Apache et jEdit :

```
bug/bug-data/source-code/source-code/<projet>/<projet>-<version>/
```

## Chemins

Les notebooks ont été exécutés dans Google Colab avec les données dans `MyDrive/bug/`. Pour une exécution locale, remplacer `/content/drive/MyDrive/bug/` par le dossier choisi.
