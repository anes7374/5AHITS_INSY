"""
Part 3: Vergleich der eigenen Implementierung mit TensorFlow.

Läuft NUR im TensorFlow-venv (TensorFlow unterstützt Python 3.14 nicht, siehe CLAUDE.md E-03):
    .venv-tf\\Scripts\\python compare_tensorflow.py

Was verglichen wird (Angabe Part 3, Step 3): Vorhersagen auf denselben Eingaben, nach gleich
vielen Epochen mit gleicher Architektur; Genauigkeit, Trainingsdauer und Konvergenz.

Netze:
    XOR         2-2-1   (Part 3, Step 1)
    7-Segment   4-16-7  (gleiche Größe wie die eigene Implementierung in Part 2 -> fairer 1:1-Vergleich)
    7-Segment   4-10-7  (Part 3, Step 2 nennt "e.g. 10" -- die eigene Implementierung wird dafür
                         ebenfalls mit 10 Hidden-Neuronen trainiert)
Alle: Sigmoid in jeder Schicht, 20000 Epochen, Lernrate 0.1, SGD mit einem Sample pro Schritt
(batch_size 1), feste Reihenfolge der Samples.

Drei Läufe pro Netz:
    eigen          die eigene Implementierung (nn/), gleicher Seed wie in Part 2
    TF identisch   TensorFlow mit EXAKT denselben Startgewichten wie "eigen", derselben
                   Kostenfunktion J = 1/2 · Summe (A - Y)² und float64. Rechnen beide wirklich
                   dieselbe Mathematik, müssen die Ergebnisse bis auf Rundungsfehler gleich sein.
                   Das ist ein starker, unabhängiger Nachweis für die eigene Backpropagation:
                   TensorFlow berechnet die Gradienten per automatischer Differentiation.
    TF Standard    so, wie man es in TensorFlow typischerweise schreibt: Keras-Standard-
                   initialisierung (Glorot uniform), Keras-MSE = Mittelwert über die Ausgänge
                   von (Y - A)², float32. Zeigt, wie sich die üblichen Voreinstellungen auswirken.

Warum eine eigene Trainingsschleife statt model.fit(): fit() hat pro Epoche einen festen
Verwaltungsaufwand (Callbacks, Logging) von einigen Millisekunden. Bei 20000 Epochen mit nur
4 bzw. 10 Samples würde das die Messung dominieren. Die Schleife unten läuft als kompilierter
TensorFlow-Graph (tf.function) und misst damit die tatsächliche Rechenleistung von TF.
Gradienten und Update kommen trotzdem vollständig von TensorFlow (GradientTape + keras SGD).

Ergebnis: Tabelle in der Konsole und ergebnisse/tf_vergleich.json (für plots.py und die
Visualisierung).
"""

import json
import os
import time
from pathlib import Path

# TensorFlow-Infomeldungen beim Import unterdrücken (müssen VOR dem Import gesetzt werden)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")   # sonst leicht schwankende Rundung

import numpy as np                      # noqa: E402
import tensorflow as tf                 # noqa: E402
from tensorflow import keras            # noqa: E402

from auswertung import genauigkeit, mse  # noqa: E402
from datasets import SEVEN_SEG_X, SEVEN_SEG_Y, XOR_X, XOR_Y  # noqa: E402
from nn import NeuralNetwork            # noqa: E402

EPOCHS = 20000
LEARNING_RATE = 0.1
SEED = 1                 # wie train_xor.py / train_7seg.py
ZIEL_KOSTEN = 0.01       # "Konvergenz": erste Epoche, in der die Kosten darunter fallen
PLOT_SCHRITT = 10        # für die JSON-Datei nur jede 10. Epoche speichern (Dateigröße)

ERGEBNISSE = Path(__file__).resolve().parent / "ergebnisse"


# ===========================================================================
# Eigene Implementierung
# ===========================================================================

def eigenes_netz(sizes):
    """Erzeugt das Netz wie in Part 2: Seed setzen, dann NeuralNetwork bauen."""
    np.random.seed(SEED)
    return NeuralNetwork(sizes, learning_rate=LEARNING_RATE)


def trainiere_eigen(net, X, Y):
    """Trainiert mit der eigenen Implementierung. Rückgabe: (Kostenverlauf, Sekunden)."""
    start = time.perf_counter()
    verlauf = net.train(X, Y, epochs=EPOCHS)
    return verlauf, time.perf_counter() - start


# ===========================================================================
# TensorFlow
# ===========================================================================

def baue_tf_modell(sizes, dtype, startnetz=None):
    """
    Baut ein Keras-Modell mit Sigmoid in jeder Schicht.

    sizes:     z.B. [4, 16, 7]
    dtype:     "float64" oder "float32"
    startnetz: eigenes NeuralNetwork, dessen Gewichte übernommen werden ("TF identisch"),
               oder None für die Keras-Standardinitialisierung ("TF Standard")

    Umrechnung der Gewichte: In nn/ ist W[l] als Zeilen pro Neuron gespeichert, Form
    (Neuronen, Eingänge), und z = W·x + b. Keras rechnet z = x·K + b mit einer Matrix K der
    Form (Eingänge, Neuronen). Also K = W transponiert.
    """
    schichten = [keras.Input(shape=(sizes[0],), dtype=dtype)]
    for size in sizes[1:]:
        schichten.append(keras.layers.Dense(size, activation="sigmoid", dtype=dtype))
    modell = keras.Sequential(schichten)

    if startnetz is not None:
        for dense, layer in zip(modell.layers, startnetz.layers):
            W = np.array([n.weights for n in layer.neurons])       # (Neuronen, Eingänge)
            b = np.array([n.bias for n in layer.neurons])          # (Neuronen,)
            dense.set_weights([W.T.astype(dtype), b.astype(dtype)])
    return modell


def kosten_wie_eigen(y, a):
    """J = 1/2 · Summe (A - Y)² -- dieselbe Kostenfunktion wie nn/network.py."""
    return 0.5 * tf.reduce_sum(tf.square(a - y))


def tf_lauf(sizes, X, Y, variante, startnetz=None):
    """
    Ein kompletter TensorFlow-Lauf (Aufbau, Kompilieren, Training).

    variante: "identisch" oder "standard" (siehe Dateikopf)
    Rückgabe: (modell, Kostenverlauf, Trainingssekunden ohne Kompilierzeit)

    Training: SGD mit batch_size 1 in fester Reihenfolge. Der Kostenverlauf misst IMMER
    J = 1/2 · Summe (A - Y)² (gemittelt über die Samples, vor dem jeweiligen Update) --
    genau wie die eigene Implementierung, damit die Kurven direkt vergleichbar sind, auch
    wenn "TF Standard" nach der Keras-MSE optimiert.
    """
    dtype = "float64" if variante == "identisch" else "float32"
    keras.utils.set_random_seed(SEED)   # macht die Standard-Initialisierung reproduzierbar
    keras.backend.set_floatx(dtype)
    modell = baue_tf_modell(sizes, dtype, startnetz if variante == "identisch" else None)
    verlust = kosten_wie_eigen if variante == "identisch" else keras.losses.MeanSquaredError()

    X_t = tf.constant(X, dtype=dtype)
    Y_t = tf.constant(Y, dtype=dtype)
    optimizer = keras.optimizers.SGD(learning_rate=LEARNING_RATE)
    n = len(X)

    @tf.function
    def epoche():
        gesamt = tf.constant(0.0, dtype=dtype)
        for i in tf.range(n):
            x = X_t[i:i + 1]
            y = Y_t[i:i + 1]
            with tf.GradientTape() as tape:
                a = modell(x, training=True)
                loss = verlust(y, a)
            gesamt += kosten_wie_eigen(y, a)
            grads = tape.gradient(loss, modell.trainable_variables)
            optimizer.apply_gradients(zip(grads, modell.trainable_variables))
        return gesamt / n

    # Kompilieren ohne zu trainieren: get_concrete_function baut den Graphen, führt ihn
    # aber nicht aus. So verändert die Kompilierung keine Gewichte und zählt nicht zur Zeit.
    epoche.get_concrete_function()

    start = time.perf_counter()
    verlauf = [float(epoche()) for _ in range(EPOCHS)]
    dauer = time.perf_counter() - start
    return modell, verlauf, dauer


# ===========================================================================
# Auswertung
# ===========================================================================

def erste_epoche_unter(verlauf, grenze=ZIEL_KOSTEN):
    """Erste Epoche (0-basiert), in der die Kosten < grenze sind; None wenn nie."""
    for epoche, kosten in enumerate(verlauf):
        if kosten < grenze:
            return epoche
    return None


def zusammenfassung(name, ausgaben, Y, verlauf, dauer):
    """Kennzahlen eines Laufs als Dictionary (für Tabelle und JSON)."""
    bits_ok, bits, samples_ok, samples = genauigkeit(ausgaben, Y)
    return {
        "name": name,
        "vorhersagen": np.asarray(ausgaben, dtype=float).tolist(),
        "mse": mse(ausgaben, Y),
        "bits_ok": bits_ok, "bits": bits,
        "samples_ok": samples_ok, "samples": samples,
        "dauer_s": dauer,
        "epoche_unter_ziel": erste_epoche_unter(verlauf),
        "kosten": verlauf[::PLOT_SCHRITT],
    }


def vergleiche(titel, sizes, X, Y):
    """Trainiert eigen / TF identisch / TF Standard und druckt die Vergleichstabelle."""
    print(f"\n{'=' * 78}\n{titel}: {' -> '.join(map(str, sizes))}, {EPOCHS} Epochen, "
          f"alpha {LEARNING_RATE}\n{'=' * 78}")

    net = eigenes_netz(sizes)
    # Startgewichte für "TF identisch" sichern, BEVOR das eigene Netz trainiert wird
    startnetz = NeuralNetwork.from_dict(net.to_dict())

    verlauf, dauer = trainiere_eigen(net, X, Y)
    eigen = zusammenfassung("eigen", [net.predict(x) for x in X], Y, verlauf, dauer)
    print(f"  eigen         fertig in {dauer:6.1f} s")

    laeufe = [eigen]
    for variante in ("identisch", "standard"):
        modell, verlauf, dauer = tf_lauf(sizes, X, Y, variante, startnetz)
        dtype = "float64" if variante == "identisch" else "float32"
        ausgaben = modell(tf.constant(X, dtype=dtype)).numpy()
        laeufe.append(zusammenfassung(f"TF {variante}", ausgaben, Y, verlauf, dauer))
        print(f"  TF {variante:<10} fertig in {dauer:6.1f} s")

    # --- Tabelle der Kennzahlen
    print(f"\n  {'Lauf':<14}{'MSE':>11}{'Bits':>9}{'Samples':>10}{'Dauer':>9}"
          f"{'Epoche J<0.01':>15}")
    for lauf in laeufe:
        ep = lauf["epoche_unter_ziel"]
        print(f"  {lauf['name']:<14}{lauf['mse']:>11.6f}"
              f"{lauf['bits_ok']:>5}/{lauf['bits']:<3}{lauf['samples_ok']:>6}/{lauf['samples']:<3}"
              f"{lauf['dauer_s']:>8.1f}s{('nie' if ep is None else str(ep)):>15}")

    # --- Vorhersagen nebeneinander (bei 7-Segment: Segment a als Beispiel und Maximum)
    eig = np.array(eigen["vorhersagen"])
    ide = np.array(laeufe[1]["vorhersagen"])
    std = np.array(laeufe[2]["vorhersagen"])
    print(f"\n  Vorhersagen je Eingabe (bei mehreren Ausgängen: der erste Ausgang)")
    print(f"  {'Eingabe':<16}{'Soll':>6}{'eigen':>12}{'TF identisch':>15}{'TF Standard':>14}")
    for x, y, e, i, s in zip(X, Y, eig, ide, std):
        print(f"  {str(x):<16}{y[0]:>6}{e[0]:>12.6f}{i[0]:>15.6f}{s[0]:>14.6f}")

    abweichung = float(np.max(np.abs(eig - ide)))
    print(f"\n  max. |eigen - TF identisch| über alle Ausgänge: {abweichung:.2e}")
    return {"titel": titel, "layer_sizes": sizes, "laeufe": laeufe,
            "abweichung_eigen_tf_identisch": abweichung}


def main():
    print(f"TensorFlow {tf.__version__}, Keras {keras.__version__}, numpy {np.__version__}")
    ergebnisse = [
        vergleiche("XOR", [2, 2, 1], XOR_X, XOR_Y),
        vergleiche("7-Segment (Größe aus Part 2)", [4, 16, 7], SEVEN_SEG_X, SEVEN_SEG_Y),
        vergleiche("7-Segment (Größe aus Part 3)", [4, 10, 7], SEVEN_SEG_X, SEVEN_SEG_Y),
    ]
    ERGEBNISSE.mkdir(exist_ok=True)
    pfad = ERGEBNISSE / "tf_vergleich.json"
    with open(pfad, "w", encoding="utf-8") as f:
        json.dump({"tensorflow": tf.__version__, "epochs": EPOCHS,
                   "learning_rate": LEARNING_RATE, "plot_schritt": PLOT_SCHRITT,
                   "vergleiche": ergebnisse}, f)
    print(f"\nGespeichert: {pfad.relative_to(pfad.parent.parent)}")


if __name__ == "__main__":
    main()
