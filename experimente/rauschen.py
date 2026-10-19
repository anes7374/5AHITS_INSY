"""
Phase 8, Punkt 4: Rausch-Robustheit des 7-Segment-Decoders (optionale Erweiterung, Seite 11:
"Try training on noisy input data (e.g., flipped bits)").

Drei Fragen:
  A) Wie robust ist das trainierte Netz aus models/7seg.json gegen analoges Rauschen auf den
     Eingängen?  Eingang = Bit + Normalverteilung(0, sigma), z.B. 1 -> 0.87, 0 -> 0.21.
  B) Hilft es, MIT Rauschen zu trainieren? Je 10 Netze (Seeds 0-9) ohne Rauschen und mit
     sigma_train = 0.2 bzw. 0.4 trainiert (in jeder Epoche frisches Rauschen), danach wie A
     getestet.
  C) Gekippte Bits beim Training: Mit Wahrscheinlichkeit p wird ein Eingangsbit umgedreht,
     der Sollwert bleibt der der ursprünglichen Ziffer.
     Wichtig zum Verständnis: Bei 4 Bit ist JEDES Muster eine gültige Zahl. Ein gekipptes Bit
     macht aus der 3 (0011) z.B. die 7 (0111) -- das Netz bekommt also widersprüchliche
     Beispiele ("0111 soll wie 3 aussehen" und "0111 soll wie 7 aussehen"). Robustheit gegen
     gekippte Bits ist bei dieser Kodierung grundsätzlich unmöglich; gemessen wird, wie viel
     Genauigkeit auf sauberen Daten das kostet.

Ergebnis: Tabellen, ergebnisse/rauschen.json und ergebnisse/rauschen.png
Ausführen: python experimente/rauschen.py      (Laufzeit ca. 2 Minuten)
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from parallel import ERGEBNISSE, PROJEKT, ParallelNetze  # noqa: E402
from datasets import SEVEN_SEG_X, SEVEN_SEG_Y  # noqa: E402
from nn import NeuralNetwork  # noqa: E402

X = np.array(SEVEN_SEG_X, dtype=float)
Y = np.array(SEVEN_SEG_Y, dtype=float)
SIGMAS = np.round(np.arange(0, 0.61, 0.05), 2)
ZIEHUNGEN = 400          # verrauschte Varianten pro Ziffer und sigma
EPOCHEN = 20000
SEEDS = range(10)


def genauigkeit_unter_rauschen(netze, sigma, rng):
    """
    Bit- und Ziffern-Genauigkeit unter Gaußschem Eingangsrauschen.
    Rückgabe: (Bit-Accuracy, Ziffern-Accuracy), je gemittelt über alle Netze und Ziehungen.
    """
    bits_ok = ziffern_ok = 0.0
    for d in range(10):
        for _ in range(ZIEHUNGEN):
            x = X[d] + rng.normal(0, sigma, size=(netze.N, 4))
            ist = netze.forward(x)[0][-1] > 0.5           # Form (N, 7)
            ok = ist == (Y[d] > 0.5)
            bits_ok += ok.mean()
            ziffern_ok += ok.all(axis=1).mean()
    n = 10 * ZIEHUNGEN
    return bits_ok / n, ziffern_ok / n


def trainiere(sigma=0.0, flip=0.0, seed_rng=123):
    """
    Trainiert 10 Netze 4-16-7 (Seeds 0-9) mit verrauschten Eingaben.
    sigma: Standardabweichung des analogen Rauschens; flip: Kipp-Wahrscheinlichkeit je Bit.
    In jeder Epoche wird neues Rauschen gezogen -- das Netz sieht nie zweimal dieselbe Eingabe.
    """
    rng = np.random.default_rng(seed_rng)
    netze = ParallelNetze.aus_seeds([4, 16, 7], SEEDS)
    for _ in range(EPOCHEN):
        for d in range(10):
            x = np.broadcast_to(X[d], (netze.N, 4)).copy()
            if flip:
                kippen = rng.random((netze.N, 4)) < flip
                x[kippen] = 1 - x[kippen]
            if sigma:
                x += rng.normal(0, sigma, size=x.shape)
            netze.sgd_schritt(x, Y[d], 0.1)
    return netze


def main():
    rng = np.random.default_rng(0)
    modell = ParallelNetze([NeuralNetwork.load(PROJEKT / "models" / "7seg.json")])

    print("A/B) Bit-Genauigkeit unter Eingangsrauschen (sigma = Standardabweichung)")
    varianten = {"models/7seg.json": modell}
    for s in (0.0, 0.2, 0.4):
        print(f"   trainiere 10 Netze mit sigma_train = {s} ...")
        varianten[f"trainiert mit sigma = {s}"] = trainiere(sigma=s)
    kurven = {}
    print(f"\n   {'sigma':>6} " + "".join(f"{n:>28}" for n in varianten))
    for sigma in SIGMAS:
        zeile = []
        for name, netze in varianten.items():
            bit, ziffer = genauigkeit_unter_rauschen(netze, sigma, rng)
            kurven.setdefault(name, []).append((float(sigma), bit, ziffer))
            zeile.append(f"{bit * 100:6.1f} % Bits {ziffer * 100:5.1f} % Z.")
        print(f"   {sigma:>6.2f} " + "".join(f"{z:>28}" for z in zeile))

    print("\nC) Training mit gekippten Eingangsbits -- Genauigkeit auf SAUBEREN Daten")
    kipp = []
    for p in (0.0, 0.05, 0.1, 0.2):
        netze = trainiere(flip=p)
        bit, ziffer = genauigkeit_unter_rauschen(netze, 0.0, rng)
        kipp.append({"p": p, "bits": bit, "ziffern": ziffer})
        print(f"   p = {p:4.2f}:  {bit * 100:5.1f} % Bits,  {ziffer * 100:5.1f} % Ziffern richtig")

    with open(ERGEBNISSE / "rauschen.json", "w", encoding="utf-8") as f:
        json.dump({"ziehungen": ZIEHUNGEN, "kurven": kurven, "gekippte_bits": kipp}, f, indent=1)
    zeichne(kurven)
    print("\nGespeichert: ergebnisse/rauschen.json, ergebnisse/rauschen.png")


def zeichne(kurven):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=130, sharex=True)
    farben = ["#141b22", "#e2441c", "#11806a", "#b4307a"]
    for (name, werte), farbe in zip(kurven.items(), farben):
        s, bit, ziffer = np.array(werte).T
        axes[0].plot(s, bit * 100, "o-", ms=3, color=farbe, label=name)
        axes[1].plot(s, ziffer * 100, "o-", ms=3, color=farbe, label=name)
    axes[0].set_ylabel("Segmente richtig (%)")
    axes[1].set_ylabel("Ziffern komplett richtig (%)")
    for ax in axes:
        ax.set_xlabel("Rauschen sigma auf den Eingängen")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(fontsize=8)
    fig.suptitle("7-Segment-Decoder unter Eingangsrauschen", fontsize=11)
    fig.tight_layout()
    fig.savefig(ERGEBNISSE / "rauschen.png")


if __name__ == "__main__":
    main()
