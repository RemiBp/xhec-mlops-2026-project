"""Execute both coursework notebooks with the locked project kernel."""

import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    nbformat.write(notebook, path)
    print(f"Executed {path.name}")
