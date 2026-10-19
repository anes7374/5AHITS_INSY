"""
Phase 8, Punkt 6: Vektorisierte Variante als Benchmark gegen die Neuron-für-Neuron-Implementierung.

nn/ bildet die Angabe 1:1 ab: Objekte Neuron -> Layer -> NeuralNetwork, jedes Neuron rechnet
sein eigenes Skalarprodukt. Das ist gut nachvollziehbar, aber in Python langsam, weil jede
Schicht eine Python-Schleife über ihre Neuronen macht.

VektorNetz unten rechnet exakt dieselbe Mathematik in Matrixform (wie in der Angabe notiert):
    Z = W·a + b   als EIN Matrix-Vektor-Produkt pro Schicht
    delta, Update  ebenfalls als ganze Vektoren/Matrizen (np.outer statt Schleife)

Gemessen wird die Zeit für das 7-Segment-Training (4-16-7, 20000 Epochen) und XOR, und
nachgewiesen, dass beide Varianten dasselbe Ergebnis liefern.

Ausführen: python experimente/vektorisiert.py      (Laufzeit ca. 1 Minute)
"""

import json
import time

import numpy as np

from parallel import ERGEBNISSE  # noqa: F401  (setzt auch den Suchpfad auf den Projektordner)
from datasets import SEVEN_SEG_X, SEVEN_SEG_Y, XOR_X, XOR_Y
from nn import NeuralNetwork
from nn.activations import get as aktivierung


class VektorNetz:
    """Dasselbe Netz wie nn.NeuralNetwork, aber jede Schicht als Matrix W (Neuronen × Eingänge)."""

    def __init__(self, net):
        """Übernimmt die Gewichte eines nn.NeuralNetwork (Kopie)."""
        self.W = [np.array([n.weights for n in L.neurons]) for L in net.layers]
        self.b = [np.array([n.bias for n in L.neurons]) for L in net.layers]
        self.acts = [aktivierung(L.activation_name) for L in net.layers]

    def forward(self, x):
        """Rückgabe: (A, Z) -- A[0] = x, A[l] = f(W[l]·A[l-1] + b[l])"""
        A, Z = [np.asarray(x, dtype=float)], []
        for W, b, (f, _) in zip(self.W, self.b, self.acts):
            Z.append(W @ A[-1] + b)
            A.append(f(Z[-1]))
        return A, Z

    def predict(self, x):
        return self.forward(x)[0][-1]

    def train(self, X, Y, epochs, lr=0.1):
        """SGD, feste Reihenfolge -- wie NeuralNetwork.train. Rückgabe: Kostenverlauf."""
        X = [np.asarray(x, dtype=float) for x in X]
        Y = [np.asarray(y, dtype=float) for y in Y]
        verlauf = []
        for _ in range(epochs):
            summe = 0.0
            for x, y in zip(X, Y):
                A, Z = self.forward(x)
                err = A[-1] - y                                   # dJ/dA = A - Y
                summe += 0.5 * float(err @ err)
                for l in range(len(self.W) - 1, -1, -1):
                    delta = err * self.acts[l][1](Z[l], A[l + 1])  # dJ/dZ
                    if l > 0:
                        err = self.W[l].T @ delta                  # mit den ALTEN Gewichten
                    self.W[l] -= lr * np.outer(delta, A[l])        # dJ/dW = delta · A[l-1]ᵀ
                    self.b[l] -= lr * delta
            verlauf.append(summe / len(X))
        return verlauf


def messe(name, sizes, X, Y, epochs=20000):
    """Trainiert dieselben Startgewichte mit beiden Varianten und vergleicht Zeit und Ergebnis."""
    np.random.seed(1)
    obj = NeuralNetwork(sizes)
    vek = VektorNetz(obj)

    t0 = time.perf_counter()
    v_obj = obj.train(X, Y, epochs=epochs)
    t_obj = time.perf_counter() - t0
    t0 = time.perf_counter()
    v_vek = vek.train(X, Y, epochs)
    t_vek = time.perf_counter() - t0

    diff = max(float(np.max(np.abs(obj.predict(x) - vek.predict(x)))) for x in X)
    kosten_diff = float(np.max(np.abs(np.array(v_obj) - np.array(v_vek))))
    print(f"{name:<18} objektorientiert {t_obj:6.1f} s   vektorisiert {t_vek:6.1f} s   "
          f"Faktor {t_obj / t_vek:4.1f}x   max. Abweichung Vorhersage {diff:.1e}, Kosten {kosten_diff:.1e}")
    return {"netz": name, "epochs": epochs, "objektorientiert_s": t_obj, "vektorisiert_s": t_vek,
            "faktor": t_obj / t_vek, "abweichung_vorhersage": diff, "abweichung_kosten": kosten_diff}


def main():
    ergebnisse = [
        messe("XOR 2-2-1", [2, 2, 1], XOR_X, XOR_Y),
        messe("7-Segment 4-16-7", [4, 16, 7], SEVEN_SEG_X, SEVEN_SEG_Y),
        messe("7-Segment 4-64-7", [4, 64, 7], SEVEN_SEG_X, SEVEN_SEG_Y, epochs=5000),
    ]
    with open(ERGEBNISSE / "vektorisiert.json", "w", encoding="utf-8") as f:
        json.dump(ergebnisse, f, indent=1)
    print("\nGespeichert: ergebnisse/vektorisiert.json")


if __name__ == "__main__":
    main()
