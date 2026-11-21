"""Le notebook de reproduction statistique s'exécute du début à la fin sans erreur."""
import nbformat
import pytest
from nbclient import NotebookClient

from conftest import RACINE

NOTEBOOK = RACINE / "notebooks/07_reproduction_statistique/reproduction_tableaux_15_17.ipynb"


@pytest.mark.lent
def test_notebook_07_s_execute(tmp_path):
    nb = nbformat.read(NOTEBOOK, as_version=4)
    NotebookClient(nb, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
    erreurs = [o for c in nb.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
    assert not erreurs
