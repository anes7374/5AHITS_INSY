"""
Tests für die C-Implementierung (Part 4 und Part 5).

Was hier nachgewiesen wird:
    1. Beide C-Programme kompilieren ohne Warnung (-Wall -Wextra -Werror).
    2. C aus .txt (Part 4) und C aus .h (Part 5) liefern für die Ziffern 0-9 dieselben
       Ausgaben wie Python -- Abweichung <= 1e-12 (gemessen: 2e-16, die letzte Binärstelle).
    3. Der C-Parser lehnt kaputte Dateien mit Exit-Code 2 ab, statt Unsinn zu rechnen.

Die Programme werden für den Test in ein temporäres Verzeichnis kompiliert -- die mit make
gebauten Programme in c_forward/ und c_header_init/ bleiben unberührt.
Ohne gcc werden die Tests übersprungen (Installation: siehe README.md).

Ausführen (im Projektordner):
    python -m pytest -v tests/test_c.py
"""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from auswertung import MODELS
from datasets import SEVEN_SEG_X
from nn import NeuralNetwork

PROJEKT = Path(__file__).resolve().parent.parent
C_FORWARD = PROJEKT / "c_forward"
C_HEADER = PROJEKT / "c_header_init"


def finde_gcc():
    """gcc im Suchpfad oder (Windows) in der winget-Installation von WinLibs."""
    gcc = shutil.which("gcc")
    if gcc:
        return gcc
    basis = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    treffer = sorted(basis.glob("BrechtSanders.WinLibs*/mingw64/bin/gcc.exe")) if basis.exists() else []
    return str(treffer[0]) if treffer else None


GCC = finde_gcc()
pytestmark = [
    pytest.mark.skipif(GCC is None, reason="kein gcc gefunden"),
    pytest.mark.skipif(not (MODELS / "7seg.txt").exists(), reason="erst train_7seg.py ausführen"),
]


def kompiliere(ausgabe, quellen, include=()):
    """Kompiliert mit denselben Flags wie die Makefiles, Warnungen gelten als Fehler."""
    befehl = [GCC, "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2",
              *[f"-I{p}" for p in include], "-o", str(ausgabe), *map(str, quellen), "-lm"]
    lauf = subprocess.run(befehl, capture_output=True, text=True,
                          env={**os.environ, "PATH": str(Path(GCC).parent) + os.pathsep + os.environ["PATH"]})
    assert lauf.returncode == 0, lauf.stderr
    return ausgabe


@pytest.fixture(scope="module")
def programme(tmp_path_factory):
    ordner = tmp_path_factory.mktemp("c")
    exe = ".exe" if os.name == "nt" else ""
    txt = kompiliere(ordner / f"nn_txt{exe}",
                     [C_FORWARD / "main_txt.c", C_FORWARD / "nn_forward.c", C_FORWARD / "test_7seg.c"])
    hdr = kompiliere(ordner / f"nn_header{exe}",
                     [C_HEADER / "main_header.c", C_FORWARD / "nn_forward.c", C_FORWARD / "test_7seg.c"],
                     include=[C_FORWARD, MODELS])
    return {"txt": txt, "header": hdr}


def roh(programm, *args):
    """Startet ein C-Programm mit --roh. Rückgabe: (Exit-Code, Matrix 10 x 7)."""
    lauf = subprocess.run([str(programm), *map(str, args), "--roh"], capture_output=True, text=True)
    werte = [[float(v) for v in z.split()[1:]] for z in lauf.stdout.splitlines() if z.strip()]
    return lauf.returncode, np.array(werte)


@pytest.mark.parametrize("variante", ["txt", "header"])
def test_c_rechnet_wie_python(programme, variante):
    args = [MODELS / "7seg.txt"] if variante == "txt" else []
    code, c_werte = roh(programme[variante], *args)
    assert code == 0, "C meldet falsche Segmente"
    python = np.array([NeuralNetwork.load(MODELS / "7seg.json").predict(x) for x in SEVEN_SEG_X])
    assert c_werte.shape == (10, 7)
    assert np.max(np.abs(c_werte - python)) <= 1e-12


def test_c_parser_lehnt_kaputte_dateien_ab(programme, tmp_path):
    original = (MODELS / "7seg.txt").read_text()
    faelle = {
        "fehlt.txt": None,
        "abgeschnitten.txt": "\n".join(original.splitlines()[:40]),
        "falsche_aktivierung.txt": original.replace("16 4 sigmoid", "16 4 softmax"),
        "zu_viele_zahlen.txt": original.replace("7 16 sigmoid", "7 15 sigmoid"),
    }
    for name, inhalt in faelle.items():
        pfad = tmp_path / name
        if inhalt is not None:
            pfad.write_text(inhalt)
        code, _ = roh(programme["txt"], pfad)
        assert code == 2, f"{name}: Exit-Code {code} statt 2"
