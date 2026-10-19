"""
Phase 8, Punkt 7: Loss-Kurven als Bilder (optionale Erweiterung Seite 11: "Visualize outputs
for both networks").

    ergebnisse/verlauf_eigen.png        XOR vs. 7-Segment, eigene Implementierung (Part 2)
    ergebnisse/verlauf_tensorflow.png   eigen vs. TF identisch vs. TF Standard (Part 3)

Datenquellen: models/*_training.json (von train_xor.py / train_7seg.py) und
ergebnisse/tf_vergleich.json (von compare_tensorflow.py). Fehlt eine Quelle, wird das
jeweilige Bild übersprungen.
Ausführen: python experimente/plots.py
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from parallel import ERGEBNISSE, PROJEKT  # noqa: E402

ROT, TEAL, MAGENTA, TINTE = "#e2441c", "#11806a", "#b4307a", "#141b22"


def schick(ax, titel):
    ax.set_yscale("log")
    ax.axhline(0.01, color="#999", lw=0.8, ls=":")
    ax.set_title(titel, fontsize=10)
    ax.set_xlabel("Epoche")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def verlauf_eigen():
    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=130)
    for name, farbe in (("7seg", ROT), ("xor", TEAL)):
        with open(PROJEKT / "models" / f"{name}_training.json", encoding="utf-8") as f:
            p = json.load(f)
        k = np.array(p["kosten"])
        ziel = int(np.argmax(k < 0.01))
        label = f"{'7-Segment 4-16-7' if name == '7seg' else 'XOR 2-2-1'} (J < 0,01 ab Epoche {ziel})"
        ax.plot(k, color=farbe, lw=1.8, label=label)
        ax.plot(ziel, k[ziel], "o", color=farbe, ms=4)
    schick(ax, "Kostenverlauf der eigenen Implementierung (20 000 Epochen, α = 0,1)")
    ax.set_ylabel("Kosten J = ½·Σ(A−Y)², Mittel über Samples")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ERGEBNISSE / "verlauf_eigen.png")
    print("Gespeichert: ergebnisse/verlauf_eigen.png")


def verlauf_tensorflow():
    pfad = ERGEBNISSE / "tf_vergleich.json"
    if not pfad.exists():
        print("tf_vergleich.json fehlt -- verlauf_tensorflow.png übersprungen")
        return
    with open(pfad, encoding="utf-8") as f:
        tf = json.load(f)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=130, sharey=True)
    for ax, v in zip(axes, tf["vergleiche"]):
        x = np.arange(len(v["laeufe"][0]["kosten"])) * tf["plot_schritt"]
        stile = [(TINTE, 5, 0.25, "-"), (ROT, 1.4, 1, "--"), (MAGENTA, 1.6, 1, "-")]
        for lauf, (farbe, lw, alpha, ls) in zip(v["laeufe"], stile):
            ax.plot(x, lauf["kosten"], ls, color=farbe, lw=lw, alpha=alpha, label=lauf["name"])
        schick(ax, f"{'-'.join(map(str, v['layer_sizes']))}  (eigen ↔ TF identisch: {v['abweichung_eigen_tf_identisch']:.0e})")
    axes[0].set_ylabel("Kosten J")
    axes[-1].legend(fontsize=8)
    fig.suptitle(f"Eigene Implementierung gegen TensorFlow {tf['tensorflow']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(ERGEBNISSE / "verlauf_tensorflow.png")
    print("Gespeichert: ergebnisse/verlauf_tensorflow.png")


if __name__ == "__main__":
    verlauf_eigen()
    verlauf_tensorflow()
