"""
esegui_notebook.py
==================
Esegue agent.ipynb in-place popolando gli output delle celle (incluse le
celle di test con il reasoning chain-of-thought e le risposte dell'agente).

Salta automaticamente:
  - la cella di pip install (le dipendenze sono gia' installate)
  - la cella finale `!streamlit run app.py` (bloccherebbe l'esecuzione)


"""

import os
import nbformat
from nbclient import NotebookClient

NOTEBOOK = "agent.ipynb"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Sottostringhe che identificano le celle da NON eseguire
SKIP_IF_CONTAINS = ("streamlit run", "pip install")


def main() -> None:
    path = os.path.join(PROJECT_DIR, NOTEBOOK)
    nb = nbformat.read(path, as_version=4)

    # Kernel con working directory = cartella del progetto (per .env e CSV)
    client = NotebookClient(
        nb,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": PROJECT_DIR}},
    )

    print(f"Esecuzione di {NOTEBOOK} ...")
    with client.setup_kernel():
        for index, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            source = cell.source
            if any(token in source for token in SKIP_IF_CONTAINS):
                print(f"  [skip]  cella {index} (contiene comando bloccante/install)")
                continue
            print(f"  [run ]  cella {index} ...")
            client.execute_cell(cell, index)

    nbformat.write(nb, path)
    print(f"Fatto. Output salvati in {NOTEBOOK}.")


if __name__ == "__main__":
    main()
