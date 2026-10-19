"""
Phase 8, Punkt 5: Seed-Studie XOR -- ab welcher Netzgröße verschwinden die lokalen Minima?

Hintergrund (CLAUDE.md, E-11): Bei XOR 2-2-1 bleiben 7 von 100 Läufen mit "klein" und 24 von
100 mit "xavier" dauerhaft in einem lokalen Minimum hängen. Hier wird das systematisch für
2, 3 und 4 Hidden-Neuronen × beide Initialisierungen gemessen.

Pro Konfiguration: Seeds 0-99, Lernrate 0.1, SGD wie in der Angabe. Ausgewertet wird nach
20000 Epochen (Vorgabe der Angabe) und nach 80000 Epochen: Wer nach 80000 Epochen noch
falsch liegt, hängt praktisch dauerhaft fest.

Ergebnis: Tabelle in der Konsole, ergebnisse/seed_studie.json und ergebnisse/seed_studie.png
Ausführen: python experimente/seed_studie.py      (Laufzeit ca. 2 Minuten)
"""

import json
import time

import matplotlib
matplotlib.use("Agg")   # ohne Fenster, nur in Dateien zeichnen
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from parallel import ERGEBNISSE, ParallelNetze, auswerten, pruefe_gleichwertigkeit  # noqa: E402
from datasets import XOR_X, XOR_Y  # noqa: E402

SEEDS = range(100)
HIDDEN = [2, 3, 4]
INITS = ["klein", "xavier"]
EPOCHEN_ANGABE = 20000
EPOCHEN_LANG = 80000


def main():
    print(f"Gleichwertigkeit mit nn/: max. Abweichung {pruefe_gleichwertigkeit((2, 2, 1), epochs=2000):.1e}\n")
    ergebnisse = []
    print(f"{'Hidden':>6} {'Init':>7} | {'richtig@20k':>11} {'sauber@20k':>10} {'Median J<0.01':>13} | "
          f"{'richtig@80k':>11} {'hängt fest':>10}   Zeit")
    for hidden in HIDDEN:
        for init in INITS:
            start = time.perf_counter()
            netze = ParallelNetze.aus_seeds([2, hidden, 1], SEEDS, init=init)
            v1 = netze.train(XOR_X, XOR_Y, EPOCHEN_ANGABE)
            r1 = auswerten(netze, XOR_X, XOR_Y, v1)
            v2 = netze.train(XOR_X, XOR_Y, EPOCHEN_LANG - EPOCHEN_ANGABE)   # weitertrainieren
            r2 = auswerten(netze, XOR_X, XOR_Y, np.vstack([v1, v2]))
            ok_ep = r1["epoche_ziel"][r1["epoche_ziel"] >= 0]
            eintrag = {
                "hidden": hidden, "init": init,
                "richtig_20k": int(r1["richtig"].sum()), "sauber_20k": int(r1["sauber"].sum()),
                "median_epoche_j001": float(np.median(ok_ep)) if len(ok_ep) else None,
                "richtig_80k": int(r2["richtig"].sum()),
                "haengt_fest": int((~r2["richtig"]).sum()),
                # Kosten der hängenden Läufe: zeigt, in welchem lokalen Minimum sie stecken
                "kosten_haengend": sorted(round(float(k), 4) for k in r2["kosten_final"][~r2["richtig"]]),
            }
            ergebnisse.append(eintrag)
            med = "–" if eintrag["median_epoche_j001"] is None else f"{eintrag['median_epoche_j001']:.0f}"
            print(f"{hidden:>6} {init:>7} | {eintrag['richtig_20k']:>11} {eintrag['sauber_20k']:>10} "
                  f"{med:>13} | {eintrag['richtig_80k']:>11} {eintrag['haengt_fest']:>10}   "
                  f"{time.perf_counter() - start:.0f} s")

    ERGEBNISSE.mkdir(exist_ok=True)
    with open(ERGEBNISSE / "seed_studie.json", "w", encoding="utf-8") as f:
        json.dump({"seeds": len(SEEDS), "learning_rate": 0.1, "ergebnisse": ergebnisse}, f, indent=1)
    zeichne(ergebnisse)
    print(f"\nGespeichert: ergebnisse/seed_studie.json, ergebnisse/seed_studie.png")


def zeichne(ergebnisse):
    """Balken: Erfolgsquote nach 20000 und 80000 Epochen je Konfiguration."""
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=130)
    farben = {"klein": "#e2441c", "xavier": "#11806a"}
    x = np.arange(len(HIDDEN))
    for i, init in enumerate(INITS):
        daten = [e for e in ergebnisse if e["init"] == init]
        pos = x + (i - 0.5) * 0.38
        ax.bar(pos, [e["richtig_20k"] for e in daten], 0.36, color=farben[init], alpha=0.45,
               label=f"{init}: richtig nach 20 000 Epochen")
        ax.bar(pos, [e["richtig_80k"] for e in daten], 0.36, fill=False, edgecolor=farben[init],
               linewidth=1.6, label=f"{init}: richtig nach 80 000 Epochen")
        for p, e in zip(pos, daten):
            ax.text(p, e["richtig_80k"] + 1.5, str(e["richtig_20k"]), ha="center", fontsize=8)
    ax.set_xticks(x, [f"2-{h}-1" for h in HIDDEN])
    ax.set_ylim(0, 112)
    ax.set_ylabel("Läufe mit 4/4 richtig (von 100 Seeds)")
    ax.set_title("XOR: Erfolgsquote nach Netzgröße und Startgewichten")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ERGEBNISSE / "seed_studie.png")


if __name__ == "__main__":
    main()
