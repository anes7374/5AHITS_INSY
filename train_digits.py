"""
Phase 9 (Erweiterung): handgeschriebene Ziffern erkennen -- Grundlage für das Zeichenfeld
der Visualisierung.

Aufgabe:   Graustufenbild einer handgeschriebenen Ziffer -> welche Ziffer (0-9)?
Daten:     MNIST, 60 000 Trainings- und 10 000 Testbilder von rund 500 Schreibern, 28×28 Pixel.
           Für das Netz auf 14×14 verkleinert (je 2×2 Pixel gemittelt): 196 statt 784 Eingänge,
           ~4× schnelleres Training, und die Hidden-Neuronen bleiben als Bild erkennbar.
           Training, Backpropagation und Vorhersage laufen vollständig über das eigene Paket nn/.
Netz:      196 Eingänge (Pixel) -> 64 Hidden (Sigmoid) -> 10 Ausgänge (Softmax)
Kosten:    Cross-Entropy  J = -log(Wahrscheinlichkeit der richtigen Ziffer)
Training:  SGD, alpha = 0.05, Reihenfolge jede Epoche neu gemischt, 15 Epochen

Warum Softmax statt 10 × Sigmoid: Die 10 Ausgänge schließen sich gegenseitig aus. Softmax
macht daraus eine Verteilung (Summe 1), die Seite kann sie direkt als "zu 97 % eine 3" zeigen.
Cross-Entropy passt dazu: Ihr Gradient an der Ausgabe ist einfach A - Y (siehe
NeuralNetwork.backward) und bleibt auch bei sehr falschen Ausgaben groß.

Warum MNIST (vorher: UCI-Ziffern 8×8, 1797 Bilder): Mit der Maus gezeichnete Ziffern wurden
nicht zuverlässig erkannt. MNIST hat 40× mehr Bilder von viel mehr Schreibern, und seine
Aufbereitung ist genau dokumentiert -- Ziffer auf 20×20 Pixel skaliert, dann mit dem
Schwerpunkt in die Mitte eines 28×28-Felds gesetzt. Genau so bereitet die Seite die Zeichnung
auf (drawingToPixels in visualisierung/vorlage.html). Begründung und Messwerte: CLAUDE.md, E-36.

Daten holen (einmalig, ~11 MB, landet in ~/.keras/datasets/mnist.npz):
           .venv-tf\\Scripts\\python -c "from tensorflow import keras; keras.datasets.mnist.load_data()"
Export:    models/digits.json            das Netz (Part 1, Step 4 -- gleiches Format)
           models/digits_training.json   Seed, Dauer, Kostenverlauf, Testgenauigkeit,
                                         Verwechslungsmatrix, 500 Testbilder für die Seite
Ausführen: py -3.14 train_digits.py     (~5 Minuten)
"""

import time
from pathlib import Path

import numpy as np

from auswertung import MODELS, speichere_protokoll
from nn import NeuralNetwork

SEITE = 14              # Bildgröße für das Netz: 14×14 Pixel
HIDDEN = 64
EPOCHS = 15
LEARNING_RATE = 0.05
SEED = 1
TESTBILDER_FUER_SEITE = 500   # so viele Testbilder bekommt die Seite für den Knopf "Testbild"
MNIST = Path.home() / ".keras" / "datasets" / "mnist.npz"


def verkleinere(bilder):
    """
    28×28 -> 14×14: je 2×2 Pixel werden gemittelt. Pixel 0..255 -> 0..1.

    bilder:   Array (Anzahl, 28, 28), Werte 0..255
    Rückgabe: Array (Anzahl, 196), zeilenweise (Pixel [r][c] = Eintrag 14·r + c)
    Die Seite macht mit der Zeichnung genau dasselbe (erst 28×28, dann 2×2 mitteln).
    """
    n = len(bilder)
    return bilder.reshape(n, SEITE, 2, SEITE, 2).mean(axis=(2, 4)).reshape(n, SEITE * SEITE) / 255.0


def lade_daten():
    """
    Lädt MNIST aus dem Keras-Cache. Training und Test sind im Datensatz fest getrennt
    (die 10 000 Testbilder stammen von anderen Schreibern als die Trainingsbilder).

    Rückgabe: (X_train, y_train, X_test, y_test, test_roh)
              X: Array (Anzahl, 196), Werte 0..1;  y: die Ziffer 0-9
              test_roh: die Testbilder in 14×14 als ganze Zahlen 0..255 (für die Seite)
    Fehler:   SystemExit mit Anleitung, wenn die Datei fehlt.
    """
    if not MNIST.exists():
        raise SystemExit(f"{MNIST} fehlt. Einmal herunterladen (siehe Dateikopf).")
    daten = np.load(MNIST)
    X_train = verkleinere(daten["x_train"].astype(float))
    X_test = verkleinere(daten["x_test"].astype(float))
    test_roh = np.rint(X_test * 255).astype(int)
    return X_train, daten["y_train"], X_test, daten["y_test"], test_roh


def one_hot(ziffern):
    """Ziffer k -> Zielvektor mit einer 1 an Stelle k (Länge 10), passend zur Softmax."""
    Y = np.zeros((len(ziffern), 10))
    Y[np.arange(len(ziffern)), ziffern] = 1.0
    return Y


def bewerte(net, X, y):
    """Rückgabe: (Anteil richtig, Verwechslungsmatrix 10×10 [Soll][Ist], Vorhersagen)"""
    vorhersagen = np.array([int(np.argmax(net.predict(x))) for x in X])
    matrix = np.zeros((10, 10), dtype=int)
    for soll, ist in zip(y, vorhersagen):
        matrix[soll, ist] += 1
    return float(np.mean(vorhersagen == y)), matrix, vorhersagen


def main():
    X_train, y_train, X_test, y_test, test_roh = lade_daten()
    Y_train = one_hot(y_train)

    np.random.seed(SEED)
    # Xavier: Bei 196 Eingängen wären 0.1-Startgewichte zu groß (z streut mit 0.1·sqrt(Summe x²));
    # Xavier skaliert passend zur Schichtgröße.
    net = NeuralNetwork([SEITE * SEITE, HIDDEN, 10], learning_rate=LEARNING_RATE,
                        activation=["sigmoid", "softmax"], init="xavier", loss="cross_entropy")
    print(f"{net}\nTraining: {EPOCHS} Epochen, alpha {LEARNING_RATE}, Seed {SEED}, "
          f"{len(X_train)} Trainings- und {len(X_test)} Testbilder\n")

    verlauf, test_verlauf = [], []
    start = time.perf_counter()
    for epoche in range(EPOCHS):
        # Eine Epoche nach der anderen, damit nach jeder die Testgenauigkeit gemessen werden kann.
        # shuffle mischt die Reihenfolge (np.random, Seed oben -> reproduzierbar).
        verlauf += net.train(X_train, Y_train, epochs=1, shuffle=True)
        test_genau = bewerte(net, X_test, y_test)[0]
        test_verlauf.append(test_genau)
        print(f"Epoche {epoche:3d}  Kosten {verlauf[-1]:.4f}  Test {test_genau * 100:5.2f} %  "
              f"({time.perf_counter() - start:.0f} s)", flush=True)
    dauer = time.perf_counter() - start

    train_genau = bewerte(net, X_train, y_train)[0]
    test_genau, matrix, vorhersagen = bewerte(net, X_test, y_test)
    print(f"\nTraining: {train_genau * 100:.2f} %   Test: {test_genau * 100:.2f} % "
          f"({int(round(test_genau * len(y_test)))}/{len(y_test)})   Dauer {dauer:.0f} s")
    print("\nVerwechslungsmatrix (Zeile = richtige Ziffer, Spalte = Vorhersage):")
    print("     " + " ".join(f"{k:4d}" for k in range(10)))
    for k in range(10):
        print(f"  {k}  " + " ".join(f"{v:4d}" for v in matrix[k]))

    n = TESTBILDER_FUER_SEITE
    dateien = [
        net.dump_json(MODELS / "digits.json"),
        speichere_protokoll(
            MODELS / "digits_training.json",
            netz="digits", datensatz="MNIST, 14x14", layer_sizes=net.layer_sizes, seed=SEED,
            epochs=EPOCHS, learning_rate=LEARNING_RATE, dauer_s=dauer, kosten=verlauf,
            test_genauigkeit_verlauf=test_verlauf, train_genauigkeit=train_genau,
            test_genauigkeit=test_genau, test_anzahl=len(y_test), verwechslung=matrix,
            # Ein Ausschnitt der Testbilder (ganzzahlig 0..255 -> kleine Datei) für den Knopf
            # "Testbild" auf der Seite, dazu die Python-Vorhersage als Referenz
            pixel_max=255, test_bilder=test_roh[:n], test_ziffern=y_test[:n],
            test_vorhersagen=vorhersagen[:n],
        ),
    ]
    print("\nExportiert:")
    for pfad in dateien:
        print(f"  {pfad.relative_to(MODELS.parent)}")
    return net


if __name__ == "__main__":
    main()
