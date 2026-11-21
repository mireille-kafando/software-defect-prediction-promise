"""Cohérence des fichiers de résultats publiés dans results/."""
import numpy as np
import pandas as pd
import pytest

from conftest import RESULTATS

PROJETS = {"ant", "camel", "ivy", "jedit", "log4j", "lucene", "poi", "synapse", "velocity", "xalan", "xerces"}


@pytest.mark.parametrize("fichier,colonnes", [
    ("cnn1d_simple_gmean.csv", ["SG_WPDP", "SV_WPDP", "SR_WPDP", "SG_CPDP", "SV_CPDP", "SR_CPDP"]),
    ("cnn1d_dense_gmean.csv", ["SG_WPDP", "SV_WPDP", "SR_WPDP", "SG_CPDP", "SV_CPDP", "SR_CPDP"]),
    ("transformer_encodeur_gmean.csv", ["WPDP_SMOTE", "CPDP_SMOTE", "WPDP_sous_ech", "CPDP_sous_ech",
                                        "WPDP_sous_ech_outliers", "CPDP_sous_ech_outliers"]),
])
def test_gmean_par_projet(fichier, colonnes):
    df = pd.read_csv(RESULTATS / fichier, index_col="projet")
    assert set(df.index) == PROJETS
    assert list(df.columns) == colonnes
    assert df.notna().all().all()
    assert ((df > 0) & (df < 100)).all().all(), "G-Mean attendues en pourcentage, entre 0 et 100"


def test_multimodal_structure():
    m = pd.read_csv(RESULTATS / "multimodal_wpdp_cpdp.csv")
    assert set(m.projet) == PROJETS - {"ivy"}, "Ivy n'a pas de code source dans le corpus"
    assert set(m.modele) == {"Hybride", "Full Pretrained"}
    assert set(m.contexte) == {"WPDP", "CPDP"}
    assert len(m) == 10 * 2 * 2
    assert not m.duplicated(["projet", "modele", "contexte"]).any()


def test_multimodal_gmean_egale_racine_sensibilite_specificite():
    m = pd.read_csv(RESULTATS / "multimodal_wpdp_cpdp.csv")
    ecart = (m.gmean - np.sqrt(m.sensibilite * m.specificite)).abs()
    assert ecart.max() < 0.02, "La G-Mean doit valoir √(sensibilité × spécificité), à l'arrondi près"
