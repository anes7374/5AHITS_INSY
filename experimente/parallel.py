"""
Viele unabhängige Netze gleichzeitig trainieren -- Werkzeug für die Experimente in Phase 8.

Warum: Die Seed-Studie und der Aktivierungsvergleich brauchen Hunderte Trainingsläufe à
20000 Epochen. Mit der Neuron-für-Neuron-Implementierung (nn/) dauert ein 7-Segment-Lauf
~30 s, 100 Läufe also fast eine Stunde -- pro Konfiguration.

Idee: N Netze gleicher Struktur werden als Stapel von Matrizen gespeichert, z.B. die Gewichte
der Hidden-Schicht als Array der Form (N, 16, 4). Jeder Rechenschritt (Forward, Backprop,
Update) läuft dann mit einem numpy-Aufruf für alle N Netze zugleich. Die Netze beeinflussen
sich dabei NICHT -- jedes macht exakt sein eigenes SGD, Sample für Sample.

Die Mathematik ist dieselbe wie in nn/ (Formeln aus der Angabe):
    Z[l] = W[l]·A[l-1] + b[l]         A[l] = f(Z[l])
    delta_L = (A - Y) · f'(Z)         delta_l = (W[l+1]ᵀ · delta_l+1) · f'(Z[l])
    W -= alpha · delta · A[l-1]ᵀ      b -= alpha · delta
Startgewichte kommen aus nn.NeuralNetwork mit demselben Seed -- ein Lauf hier startet also
exakt wie der entsprechende Lauf mit der eigenen Implementierung.

Nachweis der Gleichwertigkeit: pruefe_gleichwertigkeit() unten vergleicht mit nn/ (wird von
den Experiment-Skripten zu Beginn aufgerufen).
"""

import sys
from pathlib import Path

import numpy as np

PROJEKT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJEKT))

from nn import NeuralNetwork  # noqa: E402
from nn.activations import get as aktivierung  # noqa: E402

ERGEBNISSE = PROJEKT / "ergebnisse"


class ParallelNetze:
    """
    N Netze gleicher Struktur, die parallel mit SGD trainiert werden.

    W[l]: Form (N, Neuronen, Eingänge)    b[l]: Form (N, Neuronen)
    """

    def __init__(self, netze):
        """
        netze: Liste von nn.NeuralNetwork gleicher Struktur und Aktivierung. Ihre
               Gewichte werden kopiert (die Originale bleiben unverändert).
        """
        self.sizes = netze[0].layer_sizes
        self.N = len(netze)
        self.acts = [aktivierung(L.activation_name) for L in netze[0].layers]
        self.W = [np.array([[n.weights for n in net.layers[l].neurons] for net in netze])
                  for l in range(len(self.sizes) - 1)]
        self.b = [np.array([[n.bias for n in net.layers[l].neurons] for net in netze])
                  for l in range(len(self.sizes) - 1)]

    @classmethod
    def aus_seeds(cls, sizes, seeds, init="klein", activation="sigmoid"):
        """Startgewichte wie  np.random.seed(s); NeuralNetwork(sizes, ...)  für jeden Seed s."""
        netze = []
        for s in seeds:
            np.random.seed(s)
            netze.append(NeuralNetwork(sizes, activation=activation, init=init))
        return cls(netze)

    def forward(self, x):
        """
        x: Eingabe (Länge sizes[0]) -- für alle Netze dieselbe -- oder Form (N, sizes[0])
        Rückgabe: (A, Z) mit A[l] Form (N, Neuronen); A[0] ist die Eingabe
        """
        a = np.broadcast_to(np.asarray(x, dtype=float), (self.N, self.sizes[0]))
        A, Z = [a], []
        for W, b, (f, _) in zip(self.W, self.b, self.acts):
            # Z[n, k] = Summe_j W[n, k, j] · a[n, j] + b[n, k]   für jedes Netz n
            z = np.einsum("nkj,nj->nk", W, a) + b
            a = f(z)
            Z.append(z)
            A.append(a)
        return A, Z

    def sgd_schritt(self, x, y, lr):
        """
        Ein SGD-Schritt für ein Sample in allen N Netzen.
        Rückgabe: Kosten J = 1/2 · Summe (A - Y)² VOR dem Update, Form (N,)
        """
        A, Z = self.forward(x)
        err = A[-1] - np.asarray(y, dtype=float)       # dJ/dA = A - Y
        kosten = 0.5 * np.sum(err ** 2, axis=1)
        for l in range(len(self.W) - 1, -1, -1):
            delta = err * self.acts[l][1](Z[l], A[l + 1])   # delta = dJ/dA · f'(z)
            if l > 0:
                # Fehler für die Vorschicht mit den ALTEN Gewichten (vor dem Update)
                err = np.einsum("nk,nkj->nj", delta, self.W[l])
            self.W[l] -= lr * np.einsum("nk,nj->nkj", delta, A[l])
            self.b[l] -= lr * delta
        return kosten

    def train(self, X, Y, epochs, lr=0.1):
        """
        Trainiert alle Netze, Samples in fester Reihenfolge (wie nn/ ohne shuffle).
        Rückgabe: Kostenverlauf, Form (epochs, N) -- mittlere Kosten pro Epoche und Netz
        """
        verlauf = np.empty((epochs, self.N))
        for e in range(epochs):
            summe = np.zeros(self.N)
            for x, y in zip(X, Y):
                summe += self.sgd_schritt(x, y, lr)
            verlauf[e] = summe / len(X)
        return verlauf

    def vorhersagen(self, X):
        """Ausgaben aller Netze für alle Eingaben, Form (N, Samples, Ausgänge)."""
        return np.stack([self.forward(x)[0][-1] for x in X], axis=1)


def auswerten(netze, X, Y, verlauf, ziel=0.01):
    """
    Kennzahlen je Netz.

    Rückgabe: dict mit Arrays der Länge N:
        richtig      alle Ausgaben richtig nach Schwellwert 0.5
        sauber       jede Ausgabe weniger als 0.1 vom Sollwert entfernt
        epoche_ziel  erste Epoche mit Kosten < ziel, -1 wenn nie
    """
    P = netze.vorhersagen(X)
    Yarr = np.asarray(Y)
    richtig = np.all((P > 0.5) == (Yarr > 0.5), axis=(1, 2))
    sauber = np.all(np.abs(P - Yarr) < 0.1, axis=(1, 2))
    unter = verlauf < ziel
    epoche = np.where(unter.any(axis=0), unter.argmax(axis=0), -1)
    return {"richtig": richtig, "sauber": sauber, "epoche_ziel": epoche, "kosten_final": verlauf[-1]}


def pruefe_gleichwertigkeit(sizes=(4, 16, 7), activation="sigmoid", epochs=300):
    """
    Vergleicht einen Lauf hier mit derselben Rechnung in nn/ (gleicher Seed).
    Rückgabe: maximale Abweichung der Vorhersagen. Erwartet: ~1e-15 (Rundung in einer
    anderen Summationsreihenfolge), keinesfalls mehr als 1e-9.
    """
    from datasets import SEVEN_SEG_X, SEVEN_SEG_Y, XOR_X, XOR_Y
    X, Y = (SEVEN_SEG_X, SEVEN_SEG_Y) if sizes[0] == 4 else (XOR_X, XOR_Y)
    np.random.seed(2)
    ref = NeuralNetwork(list(sizes), activation=activation, init="xavier")
    par = ParallelNetze([ref])
    ref.train(X, Y, epochs=epochs)
    par.train(X, Y, epochs)
    return float(np.max(np.abs(par.vorhersagen(X)[0] - np.array([ref.predict(x) for x in X]))))
