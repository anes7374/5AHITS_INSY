"""
Part 2, Step 4: den exportierten 7-Segment-Decoder laden und prüfen.

Ablauf:
    1. models/7seg.json laden
    2. Eingaben [0,0,0,0] bis [1,0,0,1] (Ziffern 0-9) vorhersagen
    3. mit den Soll-Segmenten vergleichen: Bit- und Ziffern-Accuracy
    4. mit den Vorhersagen direkt nach dem Training vergleichen (models/7seg_training.json)
       -> "Confirm that import/export retains accuracy"
    5. dasselbe für models/7seg.txt
    6. ASCII-Rendering der Ziffern zur Sichtkontrolle

Voraussetzung: vorher  python train_7seg.py
Ausführen:     python test_7seg_load.py
Exit-Code:     0 wenn alles stimmt, 1 sonst
"""

import sys

import numpy as np

from auswertung import (MODELS, binaer, genauigkeit, lade_protokoll, segment_string,
                        vorhersagen, ziffern_nebeneinander)
from datasets import SEGMENT_NAMES, SEVEN_SEG_X, SEVEN_SEG_Y
from nn import NeuralNetwork

TOLERANZ = 1e-12   # JSON/TXT speichern exakt -> erwartet ist 0, siehe test_xor_load.py


def pruefe(name, net, referenz, ausfuehrlich):
    """
    Prüft ein geladenes Netz gegen Sollwerte und Referenz-Vorhersagen.

    ausfuehrlich: True -> Tabelle mit allen 70 Rohwerten ausgeben
    Rückgabe:     True, wenn 70/70 Bits stimmen und die Abweichung <= TOLERANZ ist.
    """
    ausgaben = vorhersagen(net, SEVEN_SEG_X)
    abweichung = float(np.max(np.abs(ausgaben - referenz)))
    bits_ok, bits, ziffern_ok, ziffern = genauigkeit(ausgaben, SEVEN_SEG_Y)

    print(f"\n--- geladen aus {name}: {net}")
    if ausfuehrlich:
        kopf = "".join(f"{s:>7}" for s in SEGMENT_NAMES)
        print(f"{'Ziffer':<7}{'Eingabe':<14}{kopf}   Soll     Ist      OK")
        for ziffer, (x, y, out) in enumerate(zip(SEVEN_SEG_X, SEVEN_SEG_Y, ausgaben)):
            werte = "".join(f"{v:7.3f}" for v in out)
            ist = binaer(out)
            print(f"{ziffer:<7}{str(x):<14}{werte}   {segment_string(y)}  "
                  f"{segment_string(ist)}  {'ja' if ist == y else 'NEIN'}")
        print("\nAnzeige nach Schwellwert 0.5:\n")
        print(ziffern_nebeneinander([binaer(out) for out in ausgaben]))
        print()

    erfolg = bits_ok == bits and abweichung <= TOLERANZ
    print(f"Bits richtig: {bits_ok}/{bits}   Ziffern richtig: {ziffern_ok}/{ziffern}   "
          f"max. Abweichung zur Referenz: {abweichung:.1e}   -> {'OK' if erfolg else 'FEHLER'}")
    return erfolg


def main():
    """Lädt beide Formate und prüft sie. Rückgabe: Exit-Code."""
    protokoll = lade_protokoll(MODELS / "7seg_training.json")
    referenz = np.array(protokoll["vorhersagen"])
    print(f"Referenz: Vorhersagen direkt nach dem Training "
          f"(Seed {protokoll['seed']}, {protokoll['epochs']} Epochen)")

    ergebnisse = [
        pruefe("7seg.json", NeuralNetwork.load(MODELS / "7seg.json"), referenz, True),
        pruefe("7seg.txt", NeuralNetwork.load_txt(MODELS / "7seg.txt"), referenz, False),
    ]
    alles_ok = all(ergebnisse)
    print("\nExport/Import erhält die Genauigkeit vollständig." if alles_ok
          else "\nFEHLER: Export/Import verändert das Netz!")
    return 0 if alles_ok else 1


if __name__ == "__main__":
    sys.exit(main())
