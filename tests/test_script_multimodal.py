"""Tests du script scripts/multimodal_cpdp_hybride_fullpretrained.py.

Aucun téléchargement : CodeBERT est remplacé par un modèle à poids aléatoires et le tokeniseur par
un tokeniseur factice (voir conftest.py). Les données sont synthétiques.
"""
import joblib
import numpy as np
import pandas as pd
import pytest
import torch

from conftest import TokeniseurFactice


def lot(n=4, metriques=20):
    ids = torch.randint(3, 190, (n, 12))
    return torch.randn(n, metriques), {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


@pytest.mark.parametrize("classe", ["HybridTransBugModel", "FullPretrainedModel"])
def test_passe_avant(script_reduit, classe):
    """Full Pretrained échouait dans la version conservée (384 au lieu de 768 en entrée de CodeBERT)."""
    modele = getattr(script_reduit, classe)().eval()
    with torch.no_grad():
        sortie = modele(*lot())
    assert sortie.shape == (4, 2)
    assert torch.isfinite(sortie).all()


@pytest.mark.parametrize("classe,attendu", [("HybridTransBugModel", 130_306_050),
                                            ("FullPretrainedModel", 250_910_978)])
def test_nombre_de_parametres_avec_codebert(script_codebert, classe, attendu):
    """Avec l'architecture complète de CodeBERT et 20 métriques CK, on retrouve les tailles documentées."""
    modele = getattr(script_codebert, classe)().eval()
    with torch.no_grad():
        modele(*lot(n=2))  # initialise les LazyLinear
    assert sum(p.numel() for p in modele.parameters()) == attendu


def test_versions_triees_numeriquement(script_reduit, tmp_path):
    for v in ["1.10", "1.2", "1.9"]:
        (tmp_path / f"camel-{v}.csv").write_text("name,bug\n")
    versions = [v["version"] for v in script_reduit.get_project_versions("camel", str(tmp_path))]
    assert versions == ["1.2", "1.9", "1.10"]


def test_chemin_du_code_source(script_reduit, tmp_path):
    (tmp_path / "ant" / "ant-1.7").mkdir(parents=True)
    chemin = script_reduit.get_corresponding_code_path("/x/ant-1.7.csv", str(tmp_path))
    assert chemin == str(tmp_path / "ant" / "ant-1.7")


def test_poids_de_classes(script_reduit):
    poids = script_reduit.calculate_class_weights([0, 0, 0, 1])
    assert poids.tolist() == pytest.approx([4 / 6, 2.0])


def test_evaluation_gmean(script_reduit):
    """Prédictions fixées : sensibilité 1/2, spécificité 2/3, G-Mean = √(1/3)."""

    class Fixe(torch.nn.Module):
        def forward(self, metrics, code):
            return torch.nn.functional.one_hot(metrics[:, 0].long(), 2).float()

    predits = torch.tensor([[1.0], [0.0], [0.0], [0.0], [1.0]])
    vrais = torch.tensor([1, 1, 0, 0, 0])
    code = {"input_ids": torch.zeros(5, 2, dtype=torch.long), "attention_mask": torch.ones(5, 2)}
    jeu = script_reduit.AlignedBugDataset(predits.numpy(), code, vrais.numpy(), list("abcde"))
    _, f1, sens, spec, gm, *_ = script_reduit.evaluate(Fixe(), torch.utils.data.DataLoader(jeu, batch_size=2))
    assert (sens, spec) == pytest.approx((0.5, 2 / 3), abs=1e-6)
    assert gm == pytest.approx(np.sqrt(0.5 * 2 / 3), abs=1e-6)
    assert f1 == pytest.approx(0.5)


def corpus_synthetique(racine, rng):
    """Deux projets, trois et deux versions, 30 classes Java par version."""
    donnees, sources = racine / "bug-data", racine / "source-code"
    donnees.mkdir()
    for projet, versions in {"ant": ["1.3", "1.4", "1.5"], "camel": ["1.0", "1.2"]}.items():
        for v in versions:
            noms = [f"Classe{i}" for i in range(30)]
            df = pd.DataFrame(rng.normal(size=(30, 20)) * 5 + 10, columns=[f"m{j}" for j in range(20)])
            df.insert(0, "name", noms)
            df["bug"] = (rng.random(30) < 0.3).astype(int)
            df.loc[:1, "bug"] = [0, 1]
            df.to_csv(donnees / f"{projet}-{v}.csv", index=False)
            dossier = sources / projet / f"{projet}-{v}"
            dossier.mkdir(parents=True)
            for nom in noms:
                (dossier / f"{nom}.java").write_text(f"public class {nom} {{ int x = {rng.integers(100)}; }}")
    return donnees, sources


@pytest.mark.parametrize("type_modele", ["hybrid", "full_pretrained"])
def test_entrainement_de_bout_en_bout(script_reduit, monkeypatch, tmp_path, type_modele):
    """Le protocole CPDP complet s'exécute et écrit un fichier de résultats au format publié."""
    donnees, sources = corpus_synthetique(tmp_path, np.random.default_rng(0))
    sortie = tmp_path / "resultats"
    sortie.mkdir()
    C = script_reduit.Config
    monkeypatch.setattr(C, "DATA_DIR", str(donnees))
    monkeypatch.setattr(C, "SOURCE_DIR", str(sources))
    monkeypatch.setattr(C, "SAVE_PATH", f"{sortie}/")
    monkeypatch.setattr(C, "EPOCHS", 2)
    torch.manual_seed(0)

    _, resultats = script_reduit.train_model_multi_project(type_modele, "ant")

    fichier = sortie / f"training_results_{type_modele}_ant.pkl"
    assert fichier.exists()
    sauve = joblib.load(fichier)
    assert set(sauve) == {"model_type", "model_name", "target_project", "best_f1", "final_accuracy",
                          "final_sensitivity", "final_specificity", "final_gmean", "train_losses",
                          "val_losses", "train_accuracies", "val_accuracies", "f1_scores", "gmean_scores"}
    assert len(sauve["gmean_scores"]) == 2
    assert 0 <= sauve["final_gmean"] <= 1


def test_liste_des_projets_contient_synapse(script_reduit, monkeypatch):
    vus = []
    monkeypatch.setattr(script_reduit, "train_model_multi_project",
                        lambda modele, projet: vus.append(projet) or (None, {"best_f1": 0.0}))
    monkeypatch.setattr(script_reduit.joblib, "dump", lambda *a, **k: None)
    script_reduit.main_multi_project()
    assert vus[::2] == ["ant", "camel", "jedit", "log4j", "lucene", "poi", "synapse", "velocity", "xalan", "xerces"]


def test_tokeniseur_factice_deterministe():
    t = TokeniseurFactice()
    assert torch.equal(t(["a", "b"])["input_ids"], t(["a", "b"])["input_ids"])
