"""
Phase 8, Punkt 2: Konvergenzvergleich Sigmoid / tanh / ReLU (Bonus-Aufgabe aus Part 1).

Variiert wird die Aktivierung der HIDDEN-Schicht. Die Ausgabeschicht bleibt Sigmoid, weil die
Ziele 0/1 sind und die Ausgabe als "Bit an/aus" gelesen wird (tanh lieferte -1..1, ReLU wäre
nach oben offen).

Netze aus der Angabe: XOR 2-2-1 (100 Seeds) und 7-Segment 4-16-7 (50 Seeds), je mit beiden
Initialisierungen, 20000 Epochen, Lernrate 0.1, SGD.
Gemessen: Anteil richtiger Läufe, Median der Epoche, ab der J < 0.01, Kosten am Ende.

Ergebnis: Tabelle, ergebnisse/aktivierungen.json und ergebnisse/aktivierungen.png
Ausführen: python experimente/aktivierungen.py      (Laufzeit ca. 3 Minuten)
"""

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from parallel import ERGEBNISSE, ParallelNetze, auswerten, pruefe_gleichwertigkeit  # noqa: E402
from datasets import SEVEN_SEG_X, SEVEN_SEG_Y, XOR_X, XOR_Y  # noqa: E402

EPOCHEN = 20000
AKTIVIERUNGEN = ["sigmoid", "tanh", "relu"]
INITS = ["klein", "xavier"]
AUFGABEN = [
    ("XOR 2-2-1", [2, 2, 1], XOR_X, XOR_Y, range(100)),
    ("7-Segment 4-16-7", [4, 16, 7], SEVEN_SEG_X, SEVEN_SEG_Y, range(50)),
]
FARBEN = {"sigmoid": "#e2441c", "tanh": "#11806a", "relu": "#b4307a"}


def main():
    for act in ("tanh", "relu"):
        print(f"Gleichwertigkeit mit nn/ ({act}): max. Abweichung "
              f"{pruefe_gleichwertigkeit(activation=[act, 'sigmoid']):.1e}")
    print()
    ergebnisse, kurven = [], {}
    print(f"{'Aufgabe':<18}{'Hidden':>8}{'Init':>8} | {'richtig':>8}{'Median J<0.01':>15}{'Median J Ende':>15}   Zeit")
    for name, sizes, X, Y, seeds in AUFGABEN:
        for act in AKTIVIERUNGEN:
            for init in INITS:
                start = time.perf_counter()
                netze = ParallelNetze.aus_seeds(sizes, seeds, init=init, activation=[act, "sigmoid"])
                verlauf = netze.train(X, Y, EPOCHEN)
                r = auswerten(netze, X, Y, verlauf)
                ep = r["epoche_ziel"][r["epoche_ziel"] >= 0]
                e = {
                    "aufgabe": name, "aktivierung": act, "init": init, "laeufe": len(seeds),
                    "richtig": int(r["richtig"].sum()),
                    "median_epoche_j001": float(np.median(ep)) if len(ep) else None,
                    "median_kosten_ende": float(np.median(r["kosten_final"])),
                }
                ergebnisse.append(e)
                # Median-Kostenkurve über alle Seeds, jede 20. Epoche (für die Grafik)
                kurven[(name, act, init)] = np.median(verlauf, axis=1)[::20]
                med = "–" if e["median_epoche_j001"] is None else f"{e['median_epoche_j001']:.0f}"
                print(f"{name:<18}{act:>8}{init:>8} | {e['richtig']:>4}/{len(seeds):<3}{med:>15}"
                      f"{e['median_kosten_ende']:>15.2e}   {time.perf_counter() - start:.0f} s")

    with open(ERGEBNISSE / "aktivierungen.json", "w", encoding="utf-8") as f:
        json.dump({"epochs": EPOCHEN, "learning_rate": 0.1, "ergebnisse": ergebnisse}, f, indent=1)
    zeichne(kurven)
    print("\nGespeichert: ergebnisse/aktivierungen.json, ergebnisse/aktivierungen.png")


def zeichne(kurven):
    """Median-Kostenverlauf je Aktivierung, links XOR, rechts 7-Segment; Init = Linienart."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=130, sharey=True)
    for ax, (name, *_rest) in zip(axes, AUFGABEN):
        for act in AKTIVIERUNGEN:
            for init, stil in (("klein", "-"), ("xavier", "--")):
                y = kurven[(name, act, init)]
                ax.plot(np.arange(len(y)) * 20, y, stil, color=FARBEN[act], lw=1.6,
                        label=f"{act}, {init}")
        ax.set_yscale("log")
        ax.axhline(0.01, color="#888", lw=0.8, ls=":")
        ax.set_title(f"{name}: Median über alle Seeds")
        ax.set_xlabel("Epoche")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Kosten J (Median)")
    axes[1].legend(fontsize=7.5)
    fig.suptitle("Aktivierung der Hidden-Schicht (Ausgabe immer Sigmoid)", fontsize=11)
    fig.tight_layout()
    fig.savefig(ERGEBNISSE / "aktivierungen.png")


if __name__ == "__main__":
    main()
