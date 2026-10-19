"""
Part 2, Step 1: XOR trainieren und exportieren.

Netz:      2 Eingänge -> 2 Hidden-Neuronen -> 1 Ausgang, Sigmoid
Training:  20000 Epochen, Lernrate alpha = 0.1, SGD (Update nach jedem Sample)
Export:    models/xor.json       Part 1, Step 4 (wird in test_xor_load.py wieder geladen)
           models/xor.txt        Part 1, Step 5 (menschenlesbar, auch für C)
           models/xor_net.h      Part 1, Step 6 (C-Header)
           models/xor_training.json   Seed, Dauer, Kostenverlauf und die Vorhersagen direkt
                                      nach dem Training -- die Referenz für test_xor_load.py

Ausführen (im Projektordner):
    python train_xor.py
"""

import time

import numpy as np

from auswertung import MODELS, genauigkeit, mse, speichere_protokoll, vorhersagen
from datasets import XOR_X, XOR_Y
from nn import NeuralNetwork

EPOCHS = 20000         # laut Angabe
LEARNING_RATE = 0.1    # laut Angabe (alpha)
SEED = 1               # fester Startpunkt -> jeder Lauf liefert dasselbe Netz


def main():
    """Trainiert, gibt die Ergebnistabelle aus und schreibt alle Exportdateien."""
    # Mit nur 2 Hidden-Neuronen hängt es vom Startpunkt ab, ob XOR gelernt wird
    # (Messung in CLAUDE.md, E-11). Seed 1 lernt XOR sauber; der Seed macht den Lauf
    # reproduzierbar, damit jeder dieselben Exportdateien bekommt.
    np.random.seed(SEED)
    net = NeuralNetwork([2, 2, 1], learning_rate=LEARNING_RATE)

    print(f"{net}\nTraining: {EPOCHS} Epochen, Lernrate {LEARNING_RATE}, Seed {SEED}\n")
    start = time.perf_counter()
    verlauf = net.train(XOR_X, XOR_Y, epochs=EPOCHS, verbose_every=2000)
    dauer = time.perf_counter() - start

    ausgaben = vorhersagen(net, XOR_X)
    print("\nErgebnisse:")
    print(f"{'Eingabe':<10}{'Soll':<6}{'Ausgabe':<12}{'gerundet':<10}OK")
    for x, y, out in zip(XOR_X, XOR_Y, ausgaben):
        gerundet = int(out[0] > 0.5)
        print(f"{str(x):<10}{y[0]:<6}{out[0]:<12.6f}{gerundet:<10}"
              f"{'ja' if gerundet == y[0] else 'NEIN'}")

    _, _, ok, gesamt = genauigkeit(ausgaben, XOR_Y)
    print(f"\n{ok}/{gesamt} korrekt, Kosten (MSE) {mse(ausgaben, XOR_Y):.6f}, "
          f"Trainingsdauer {dauer:.1f} s")

    # Export in alle drei Formate (Part 1, Step 4-6)
    titel = "XOR-Netz (Part 2, Step 1)"
    dateien = [
        net.dump_json(MODELS / "xor.json"),
        net.dump_txt(MODELS / "xor.txt", titel),
        net.dump_initfile(MODELS / "xor_net.h", titel),
        speichere_protokoll(
            MODELS / "xor_training.json",
            netz="xor", layer_sizes=net.layer_sizes, seed=SEED, epochs=EPOCHS,
            learning_rate=LEARNING_RATE, dauer_s=dauer, kosten=verlauf,
            inputs=XOR_X, targets=XOR_Y, vorhersagen=ausgaben,
        ),
    ]
    print("\nExportiert:")
    for pfad in dateien:
        print(f"  {pfad.relative_to(MODELS.parent)}")
    return net


if __name__ == "__main__":
    main()
