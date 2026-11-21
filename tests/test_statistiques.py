"""Tests statistiques du mémoire recalculés à partir de results/ (tableau 15, section 5.2.4).

Les valeurs attendues sont celles du mémoire lorsqu'elles sont reproduites, et les valeurs recalculées
documentées dans results/statistiques/README.md lorsqu'elles diffèrent du mémoire.
"""
import pandas as pd
import pytest
from scipy import stats

from conftest import RACINE, RESULTATS

simple = pd.read_csv(RESULTATS / "cnn1d_simple_gmean.csv", index_col="projet")
dense = pd.read_csv(RESULTATS / "cnn1d_dense_gmean.csv", index_col="projet")
encodeur = pd.read_csv(RESULTATS / "transformer_encodeur_gmean.csv", index_col="projet")
multimodal = pd.read_csv(RESULTATS / "multimodal_wpdp_cpdp.csv")
hybride = multimodal.query("modele == 'Hybride' and contexte == 'CPDP'").set_index("projet")["gmean"]


def friedman(*series):
    chi2, p = stats.friedmanchisquare(*series)
    n, k = len(series[0]), len(series)
    return chi2, p, chi2 / (n * (k - 1))


@pytest.mark.parametrize("contexte,attendu", [("WPDP", (0.727, 0.695, 0.033)), ("CPDP", (2.182, 0.336, 0.099))])
def test_friedman_standardisation(contexte, attendu):
    obtenu = friedman(*(simple[f"{s}_{contexte}"] for s in ("SG", "SV", "SR")))
    assert obtenu == pytest.approx(attendu, abs=1e-3)


@pytest.mark.parametrize("contexte,stat,p", [("WPDP", 28.0, 0.700), ("CPDP", 23.0, 0.413)])
def test_wilcoxon_cnn1d_simple_dense(contexte, stat, p):
    r = stats.wilcoxon(simple[f"SG_{contexte}"], dense[f"SG_{contexte}"])
    assert (r.statistic, r.pvalue) == pytest.approx((stat, p), abs=1e-3)


def test_wilcoxon_cnn1d_encodeur_cpdp():
    r = stats.wilcoxon(simple["SG_CPDP"], encodeur["CPDP_SMOTE"])
    assert r.statistic == 6.0
    assert r.pvalue == pytest.approx(0.014, abs=1e-3)


def test_friedman_cnn1d_encodeur_hybride_cpdp():
    projets = hybride.index
    chi2, p, w = friedman(simple.loc[projets, "SG_CPDP"], encodeur.loc[projets, "CPDP_SMOTE"], hybride)
    assert (chi2, p, w) == pytest.approx((7.8, 0.020, 0.39), abs=1e-3)


def test_tableau_de_reproduction_publie_a_jour():
    """Le CSV publié par le notebook 07 correspond aux calculs ci-dessus."""
    t = pd.read_csv(RACINE / "results/statistiques/reproduction/tableau15_reproduction.csv")
    assert len(t) == 8
    assert t.concorde.sum() == 5
    assert t.loc[t.comparaison == "CNN1D vs encodeur vs Hybride", "statistique"].item() == pytest.approx(7.8)
