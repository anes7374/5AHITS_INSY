"""
Einstiegspunkt aus dem Ausgangsstand -- trainiert XOR mit dem Paket nn.

Seit Phase 1 liegt der eigentliche Code im Paket nn/ (siehe Implementation.md).
Diese Datei bleibt bestehen, damit beides weiter funktioniert, was vorher ging:

    python neural_network.py                    # trainiert XOR und zeigt das Ergebnis
    from neural_network import NeuralNetwork    # Import aus anderen Skripten

Die ursprüngliche, vollständig kommentierte Fassung dieser Datei liegt unverändert in
archiv/neural_network_ausgangsstand.py -- darauf verweist Ausgangsstand.md.

Bezug zur Angabe: Part 2, Step 1 (XOR, 2-2-1, 20000 Epochen, Lernrate 0.1).
Der dort geforderte JSON-Export steht in train_xor.py (dieses Skript exportiert nichts).
"""

import numpy as np

from datasets import XOR_X, XOR_Y
# Layer und Neuron werden hier nicht benutzt, sondern nur weitergereicht:
# "from neural_network import Neuron" soll wie im Ausgangsstand funktionieren.
from nn import Layer, NeuralNetwork, Neuron  # noqa: F401

# Alte Namen weiter anbieten, damit Code, der sie importiert, nicht bricht
X = XOR_X
Y = XOR_Y

EPOCHS = 20000         # laut Angabe
LEARNING_RATE = 0.1    # laut Angabe (alpha)


def train_xor():
    """
    Trainiert ein 2-2-1-Netz auf XOR und gibt eine Ergebnistabelle aus.

    Ablauf: Seed setzen -> Netz bauen -> trainieren -> alle 4 Eingaben testen.
    Eine Ausgabe > 0.5 zählt als 1, sonst als 0 (Schwellwert).

    Rückgabe: das trainierte Netz
    """
    # Fester Seed, damit jeder Lauf dieselben Startgewichte und damit dasselbe Ergebnis
    # liefert. Mit nur 2 Hidden-Neuronen hängt es vom Startpunkt ab, ob XOR in 20000
    # Epochen gelernt wird (Messung dazu in CLAUDE.md, Eintrag zur Initialisierung).
    np.random.seed(1)

    # [2, 2, 1] = 2 Eingänge, 2 Hidden-Neuronen, 1 Output-Neuron
    net = NeuralNetwork([2, 2, 1], learning_rate=LEARNING_RATE)

    print(f"{net}\nTraining: {EPOCHS} Epochen, Lernrate {LEARNING_RATE}\n")
    # Kosten alle 2000 Epochen ausgeben -> man sieht, wie der Fehler sinkt
    net.train(X, Y, epochs=EPOCHS, verbose_every=2000)

    # Ergebnistabelle: Eingabe | Sollwert | rohe Netzausgabe | gerundet | richtig?
    print("\nErgebnisse:")
    print(f"{'Input':<10}{'Erwartet':<12}{'Ausgabe':<12}{'Gerundet':<10}{'OK'}")
    korrekt = 0
    for x, y in zip(X, Y):
        # predict liefert ein Array mit einem Eintrag -> [0] holt die Zahl heraus
        out = net.predict(x)[0]
        # Schwellwert 0.5: Sigmoid liefert nie exakt 0 oder 1, darum wird gerundet
        gerundet = int(out > 0.5)
        ok = gerundet == y[0]
        korrekt += ok   # True zählt als 1, False als 0
        print(f"{str(x):<10}{y[0]:<12}{out:<12.6f}{gerundet:<10}{'ja' if ok else 'NEIN'}")

    print(f"\n{korrekt}/{len(X)} korrekt klassifiziert")

    return net


# Nur ausführen, wenn die Datei direkt gestartet wird, nicht beim Import
if __name__ == "__main__":
    train_xor()
