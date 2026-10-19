"""
Zusätzliche Abbildungen für die Quarto-Dokumentation (die übrigen liegen schon in ergebnisse/).

Erzeugt in dokumentation/bilder/:
    sigmoid.png          Sigmoid und ihre Ableitung a·(1−a) -- Theorieteil
    ziffern_verlauf.png  Kosten und Testgenauigkeit der Ziffernerkennung pro Epoche
    ziffern_matrix.png   Verwechslungsmatrix auf den 10 000 MNIST-Testbildern
    ziffern_beispiele.png  Testbilder (14×14) mit der Vorhersage des eigenen Netzes
und kopiert die Diagramme aus ergebnisse/ hierher (Typst liest nur innerhalb dieses Ordners).

Liest nur fertige Ergebnisse (models/digits.json, models/digits_training.json) -- es wird nichts
neu trainiert. Farben wie in experimente/plots.py: Orange = eigene Implementierung/Aktivierung,
Teal = zweite Größe.

Ausführen: py -3.14 dokumentation/abbildungen.py
"""

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # ohne Fenster, nur Dateien
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HIER = Path(__file__).resolve().parent
PROJEKT = HIER.parent
sys.path.insert(0, str(PROJEKT))
from nn import NeuralNetwork  # noqa: E402

BILDER = HIER / "bilder"
ORANGE, TEAL, GRAU = "#e2441c", "#11806a", "#75818d"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 150})


def sigmoid_bild():
    z = np.linspace(-8, 8, 400)
    a = 1 / (1 + np.exp(-z))
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(z, a, color=ORANGE, lw=2.2, label=r"$\sigma(z) = 1/(1+e^{-z})$")
    ax.plot(z, a * (1 - a), color=TEAL, lw=2.2, label=r"$\sigma'(z) = a\,(1-a)$, höchstens 0,25")
    ax.axhline(0.5, color=GRAU, lw=0.8, ls=":")
    ax.set_xlabel("z  (gewichtete Summe)")
    ax.set_ylabel("Wert")
    ax.legend(loc="center right", frameon=False)
    fig.tight_layout()
    fig.savefig(BILDER / "sigmoid.png")
    plt.close(fig)


def ziffern_bilder():
    prot = json.loads((PROJEKT / "models" / "digits_training.json").read_text(encoding="utf-8"))
    net = NeuralNetwork.load(PROJEKT / "models" / "digits.json")

    # Lernkurve: Kosten (links) und Testgenauigkeit (rechts) pro Epoche
    ep = np.arange(1, prot["epochs"] + 1)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(ep, prot["kosten"], color=ORANGE, lw=2.2, marker="o", ms=3.5, label="Kosten J (Cross-Entropy)")
    ax.set_xlabel("Epoche")
    ax.set_ylabel("Kosten J", color=ORANGE)
    ax2 = ax.twinx()
    ax2.plot(ep, np.array(prot["test_genauigkeit_verlauf"]) * 100, color=TEAL, lw=2.2, marker="s", ms=3.5)
    ax2.set_ylabel("Testgenauigkeit in %", color=TEAL)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax.set_title(f"MNIST 14×14, Netz 196-64-10: {prot['test_genauigkeit'] * 100:.2f} % nach "
                 f"{prot['epochs']} Epochen", fontsize=10)
    fig.tight_layout()
    fig.savefig(BILDER / "ziffern_verlauf.png")
    plt.close(fig)

    # Verwechslungsmatrix, Diagonale ausgeblendet, damit die Fehler sichtbar werden
    m = np.array(prot["verwechslung"])
    fehler = m.copy()
    np.fill_diagonal(fehler, 0)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.imshow(fehler, cmap="Oranges")
    ax.grid(False)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, m[i, j], ha="center", va="center", fontsize=8,
                    color="black" if i != j else TEAL, fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Vorhersage des Netzes")
    ax.set_ylabel("richtige Ziffer")
    ax.set_title("Verwechslungsmatrix (Farbe = Fehler)", fontsize=10)
    fig.tight_layout()
    fig.savefig(BILDER / "ziffern_matrix.png")
    plt.close(fig)

    # Beispiel-Testbilder mit Vorhersage; falsch erkannte rot beschriftet
    bilder = np.array(prot["test_bilder"])[:32] / prot["pixel_max"]
    soll = prot["test_ziffern"][:32]
    fig, achsen = plt.subplots(4, 8, figsize=(8, 4.4))
    for ax, x, s in zip(achsen.flat, bilder, soll):
        p = net.predict(x)
        ist = int(np.argmax(p))
        ax.imshow(x.reshape(14, 14), cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"{ist} ({p[ist] * 100:.0f} %)", fontsize=8, color=TEAL if ist == s else ORANGE)
        ax.axis("off")
    fig.suptitle("MNIST-Testbilder (14×14) und Vorhersage des eigenen Netzes", fontsize=10)
    fig.tight_layout()
    fig.savefig(BILDER / "ziffern_beispiele.png")
    plt.close(fig)


def kopiere_ergebnisse():
    """
    Kopiert die vorhandenen Diagramme aus ergebnisse/ nach bilder/. Nötig, weil Typst (PDF)
    aus Sicherheitsgründen keine Dateien außerhalb des Dokumentationsordners lesen darf.
    """
    for name in ("verlauf_eigen", "verlauf_tensorflow", "seed_studie", "aktivierungen", "rauschen"):
        shutil.copyfile(PROJEKT / "ergebnisse" / f"{name}.png", BILDER / f"{name}.png")
        print(f"  dokumentation/bilder/{name}.png  (Kopie aus ergebnisse/)")


def main():
    BILDER.mkdir(exist_ok=True)
    sigmoid_bild()
    ziffern_bilder()
    for name in ("sigmoid", "ziffern_verlauf", "ziffern_matrix", "ziffern_beispiele"):
        print(f"  dokumentation/bilder/{name}.png")
    kopiere_ergebnisse()


if __name__ == "__main__":
    main()
