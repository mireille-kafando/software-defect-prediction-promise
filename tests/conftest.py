"""Configuration commune des tests.

Les tests n'utilisent ni les données PROMISE ni les poids de CodeBERT : le script multimodal est importé
avec des chemins temporaires, un CodeBERT réduit à poids aléatoires et un tokeniseur factice.
"""
import importlib
import os
import sys
import tempfile
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[1]
RESULTATS = RACINE / "results"
sys.path.insert(0, str(RACINE / "scripts"))

# Le module crée son dossier de résultats à l'import : on le dirige vers un dossier temporaire.
os.environ.setdefault("SDP_RESULTS_DIR", tempfile.mkdtemp(prefix="sdp_results_"))

# Configuration de microsoft/codebert-base (config.json sur Hugging Face).
CODEBERT = dict(vocab_size=50265, hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                intermediate_size=3072, max_position_embeddings=514, type_vocab_size=1,
                layer_norm_eps=1e-5, pad_token_id=1, bos_token_id=0, eos_token_id=2)
# Version réduite : la largeur 768 est conservée car le script la code en dur ; le vocabulaire
# couvre les identifiants 0-999 utilisés par le script pour initialiser les LazyLinear.
CODEBERT_REDUIT = dict(CODEBERT, vocab_size=1024, num_hidden_layers=1, intermediate_size=64,
                       max_position_embeddings=40)


class TokeniseurFactice:
    """Remplace RobertaTokenizer : identifiants déterministes tirés du texte, séquences courtes."""

    longueur = 16

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, textes, **kwargs):
        import torch

        ids = torch.tensor([[3 + (hash((t, i)) % 190) for i in range(self.longueur)] for t in textes])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def importer_script(monkeypatch, config_codebert):
    """Importe le script multimodal avec un CodeBERT à poids aléatoires de la configuration donnée."""
    from transformers import RobertaConfig, RobertaModel

    module = importlib.import_module("multimodal_cpdp_hybride_fullpretrained")
    cfg = RobertaConfig(**config_codebert)
    monkeypatch.setattr(module.RobertaModel, "from_pretrained",
                        classmethod(lambda cls, *a, **k: RobertaModel(cfg)), raising=False)
    monkeypatch.setattr(module, "RobertaTokenizer", TokeniseurFactice)
    return module


@pytest.fixture
def script_reduit(monkeypatch):
    return importer_script(monkeypatch, CODEBERT_REDUIT)


@pytest.fixture
def script_codebert(monkeypatch):
    return importer_script(monkeypatch, CODEBERT)
