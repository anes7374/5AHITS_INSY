"""
Part 2, Step 3: 7-Segment-Decoder trainieren und exportieren.

Aufgabe:   4-Bit-Binärzahl (Ziffer 0-9) -> 7 Ausgänge für die Segmente a-g
Netz:      4 Eingänge -> 16 Hidden-Neuronen -> 7 Ausgänge, Sigmoid
Training:  20000 Epochen, Lernrate alpha = 0.1, SGD, 10 Samples (Tabelle aus datasets.py)
Export:    models/7seg.json            Part 1, Step 4 (wird in test_7seg_load.py geladen)
           models/7seg.txt             Part 1, Step 5 (menschenlesbar; Part 4 lädt sie in C)
           models/7seg_net.h           Part 1, Step 6 (C-Header; Part 5 nutzt ihn)
           models/7seg_training.json   Seed, Dauer, Kostenverlauf, Referenz-Vorhersagen

Ausführen (im Projektordner):
    python train_7seg.py
"""

import time

import numpy as np

from auswertung import (MODELS, binaer, genauigkeit, mse, segment_string,
                        speichere_protokoll, vorhersagen, ziffern_nebeneinander)
from datasets import SEVEN_SEG_X, SEVEN_SEG_Y
from nn import NeuralNetwork

EPOCHS = 20000          # laut Angabe
LEARNING_RATE = 0.1     # laut Angabe
HIDDEN = 16             # laut Angabe ("can be adjusted")
SEED = 1                # fester Startpunkt -> reproduzierbare Exportdateien


def main():
    """Trainiert, zeigt die Ergebnisse und schreibt alle Exportdateien."""
    np.random.seed(SEED)
    net = NeuralNetwork([4, HIDDEN, 7], learning_rate=LEARNING_RATE)

    print(f"{net}\nTraining: {EPOCHS} Epochen, Lernrate {LEARNING_RATE}, Seed {SEED}, "
          f"{len(SEVEN_SEG_X)} Samples\n")
    start = time.perf_counter()
    verlauf = net.train(SEVEN_SEG_X, SEVEN_SEG_Y, epochs=EPOCHS, verbose_every=2000)
    dauer = time.perf_counter() - start

    ausgaben = vorhersagen(net, SEVEN_SEG_X)
    print("\nErgebnisse (Segmente a b c d e f g):")
    print(f"{'Ziffer':<8}{'Eingabe':<15}{'Soll':<10}{'Ist':<10}{'kleinster Abstand zu 0.5':<26}OK")
    for ziffer, (x, y, out) in enumerate(zip(SEVEN_SEG_X, SEVEN_SEG_Y, ausgaben)):
        ist = binaer(out)
        # Wie "sicher" das Netz ist: das Segment, das am nächsten an der Schwelle 0.5 liegt
        sicherheit = float(np.min(np.abs(out - 0.5)))
        print(f"{ziffer:<8}{str(x):<15}{segment_string(y):<10}{segment_string(ist):<10}"
              f"{sicherheit:<26.4f}{'ja' if ist == y else 'NEIN'}")

    bits_ok, bits, ziffern_ok, ziffern = genauigkeit(ausgaben, SEVEN_SEG_Y)
    print("\nSo würde die Anzeige aussehen:\n")
    print(ziffern_nebeneinander([binaer(out) for out in ausgaben]))
    print(f"\nBits richtig: {bits_ok}/{bits}   Ziffern richtig: {ziffern_ok}/{ziffern}   "
          f"Kosten (MSE) {mse(ausgaben, SEVEN_SEG_Y):.6f}   Trainingsdauer {dauer:.1f} s")

    titel = "7-Segment-Decoder (Part 2, Step 3)"
    dateien = [
        net.dump_json(MODELS / "7seg.json"),
        net.dump_txt(MODELS / "7seg.txt", titel),
        net.dump_initfile(MODELS / "7seg_net.h", titel),
        speichere_protokoll(
            MODELS / "7seg_training.json",
            netz="7seg", layer_sizes=net.layer_sizes, seed=SEED, epochs=EPOCHS,
            learning_rate=LEARNING_RATE, dauer_s=dauer, kosten=verlauf,
            inputs=SEVEN_SEG_X, targets=SEVEN_SEG_Y, vorhersagen=ausgaben,
        ),
    ]
    print("\nExportiert:")
    for pfad in dateien:
        print(f"  {pfad.relative_to(MODELS.parent)}")
    return net


if __name__ == "__main__":
    main()
