"""
Part 2, Step 2: das exportierte XOR-Netz laden und prüfen.

Ablauf:
    1. models/xor.json laden (Part 1, Step 4: load)
    2. dieselben 4 Eingaben vorhersagen
    3. mit den Vorhersagen vergleichen, die das Netz DIREKT NACH DEM TRAINING geliefert hat
       (gespeichert von train_xor.py in models/xor_training.json)
    4. zusätzlich dasselbe mit models/xor.txt (Part 1, Step 5) -- auch dieses Format muss
       das Netz verlustfrei enthalten

"Within tolerance" aus der Angabe: Weil JSON und TXT die Gewichte exakt speichern, muss die
Abweichung 0 sein. Die Toleranz 1e-12 ist nur Sicherheitsabstand für Rundung in numpy.

Voraussetzung: vorher  python train_xor.py
Ausführen:     python test_xor_load.py
Exit-Code:     0 wenn alles stimmt, 1 sonst (so lässt sich das Skript auch automatisch prüfen)
"""

import sys

import numpy as np

from auswertung import MODELS, genauigkeit, lade_protokoll, vorhersagen
from datasets import XOR_X, XOR_Y
from nn import NeuralNetwork

TOLERANZ = 1e-12


def pruefe(name, net, referenz):
    """
    Vergleicht die Vorhersagen eines geladenen Netzes mit der Referenz.

    Rückgabe: True, wenn die maximale Abweichung <= TOLERANZ ist und alle 4 Fälle
              richtig klassifiziert werden.
    """
    ausgaben = vorhersagen(net, XOR_X)
    abweichung = float(np.max(np.abs(ausgaben - referenz)))

    print(f"\n--- geladen aus {name}: {net}")
    print(f"{'Eingabe':<10}{'Soll':<6}{'nach Training':<16}{'nach Laden':<16}{'Differenz'}")
    for x, y, ref, out in zip(XOR_X, XOR_Y, referenz, ausgaben):
        print(f"{str(x):<10}{y[0]:<6}{ref[0]:<16.12f}{out[0]:<16.12f}{abs(out[0] - ref[0]):.1e}")

    _, _, ok, gesamt = genauigkeit(ausgaben, XOR_Y)
    erfolg = abweichung <= TOLERANZ and ok == gesamt
    print(f"Accuracy (Schwellwert 0.5): {ok}/{gesamt}   "
          f"maximale Abweichung: {abweichung:.1e} (Toleranz {TOLERANZ:.0e})   "
          f"-> {'OK' if erfolg else 'FEHLER'}")
    return erfolg


def main():
    """Lädt beide Exportformate, vergleicht sie mit der Referenz. Rückgabe: Exit-Code."""
    protokoll = lade_protokoll(MODELS / "xor_training.json")
    referenz = np.array(protokoll["vorhersagen"])
    print(f"Referenz: Vorhersagen direkt nach dem Training "
          f"(Seed {protokoll['seed']}, {protokoll['epochs']} Epochen)")

    ergebnisse = [
        pruefe("xor.json", NeuralNetwork.load(MODELS / "xor.json"), referenz),
        pruefe("xor.txt", NeuralNetwork.load_txt(MODELS / "xor.txt"), referenz),
    ]
    alles_ok = all(ergebnisse)
    print("\nExport/Import funktioniert verlustfrei." if alles_ok
          else "\nFEHLER: Export/Import verändert das Netz!")
    return 0 if alles_ok else 1


if __name__ == "__main__":
    sys.exit(main())
