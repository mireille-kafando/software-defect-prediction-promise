# Données

Les données ne sont pas incluses dans ce dépôt. Il s'agit de jeux de données publics de tiers :

1. **Métriques PROMISE** : métriques de code orienté objet (CK, McCabe, etc.) et nombre de défauts par classe, pour plusieurs versions de projets Java open source (Jureczko et Madeyski, 2010).
2. **Code source** des versions correspondantes de ces projets (projets Apache et autres projets Java open source), utilisé pour les représentations CodeBERT.

## Projets utilisés

`ant`, `camel`, `ivy` (CNN1D uniquement), `jedit`, `log4j`, `lucene`, `poi`, `synapse`, `velocity`, `xalan`, `xerces`

## Arborescence attendue

Les notebooks ont été exécutés sur Google Colab avec Google Drive monté. Ils attendent cette arborescence :

```
/content/drive/MyDrive/bug/bug-data/
├── ant/
│   ├── ant-1.3.csv
│   ├── ant-1.4.csv
│   └── ...
├── camel/
│   └── camel-1.0.csv ...
├── ...
└── source-code/source-code/
    ├── ant/ant-1.3/ ...
    └── ...
```

Pour une exécution locale, remplacer le préfixe `/content/drive/MyDrive/bug/` par le dossier où se trouvent les données, et retirer les cellules `drive.mount(...)`.

## Référence

M. Jureczko et L. Madeyski, « Towards identifying software project clusters with regard to defect prediction », *Proceedings of the 6th International Conference on Predictive Models in Software Engineering (PROMISE)*, 2010.
