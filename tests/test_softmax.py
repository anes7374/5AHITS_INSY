"""
Tests für Phase 9: Softmax-Schicht und Cross-Entropy (Grundlage der Ziffernerkennung).

Was hier nachgewiesen wird:
    1. softmax() liefert eine Wahrscheinlichkeitsverteilung und läuft auch bei riesigen z
       nicht über.
    2. Gradient-Check wie in test_kern.py, aber für die neuen Kombinationen:
       softmax + Cross-Entropy (Abkürzung dJ/dZ = A - Y), softmax + MSE (volle
       Jacobi-Matrix in Layer.backward) und sigmoid + Cross-Entropy.
    3. Die Kostenfunktion wird mit exportiert und wieder geladen; alte Exportdateien
       (ohne "loss") bleiben unverändert.
    4. Unsinnige Kombinationen werden mit einer klaren Meldung abgelehnt.

Ausführen:  py -3.14 -m pytest tests/test_softmax.py -v
"""

import copy
import json

import numpy as np
import pytest

from nn import NeuralNetwork, activations


# ===========================================================================
# 1. Softmax selbst
# ===========================================================================

def test_softmax_ist_eine_wahrscheinlichkeitsverteilung():
    a = activations.softmax(np.array([2.0, -1.0, 0.5, 3.0]))
    assert np.all(a > 0)
    assert np.sum(a) == pytest.approx(1.0, abs=1e-15)
    assert np.argmax(a) == 3           # größtes z -> größte Wahrscheinlichkeit


def test_softmax_laeuft_bei_grossen_werten_nicht_ueber():
    """Ohne das Abziehen von max(z) wäre e^1000 = inf und das Ergebnis NaN."""
    a = activations.softmax(np.array([1000.0, 999.0, -1000.0]))
    assert np.all(np.isfinite(a))
    assert a[0] == pytest.approx(1 / (1 + np.exp(-1)))


def test_softmax_schicht_gibt_summe_1_aus():
    np.random.seed(0)
    net = NeuralNetwork([5, 8, 4], activation=["sigmoid", "softmax"], init="xavier")
    out = net.predict(np.random.uniform(0, 1, 5))
    assert np.sum(out) == pytest.approx(1.0, abs=1e-15)
    # Der Cache jedes Neurons enthält danach a (nicht z) -- wie bei jeder anderen Schicht
    assert [n._last_output for n in net.layers[-1].neurons] == list(out)


# ===========================================================================
# 2. Gradient-Check (Aufbau wie in test_kern.py, Kosten über net.cost)
# ===========================================================================

def analytische_gradienten(net, x, y):
    """backward() auf einer Kopie mit alpha = 1 -> dJ/dW = W_alt - W_neu (siehe test_kern.py)."""
    kopie = copy.deepcopy(net)
    kopie.backward(kopie.forward(x), y, learning_rate=1.0)
    return [np.append(alt.weights - neu.weights, alt.bias - neu.bias)
            for la, lb in zip(net.layers, kopie.layers) for alt, neu in zip(la.neurons, lb.neurons)]


def numerische_gradienten(net, x, y, h=1e-6):
    """Zentrale Differenz über net.cost -- weiß nichts von Softmax-Ableitung oder Abkürzungen."""
    y = np.array(y, dtype=float)
    J = lambda: net.cost(net.forward(x), y)   # noqa: E731
    ergebnis = []
    for layer in net.layers:
        for neuron in layer.neurons:
            g = []
            for j in range(len(neuron.weights) + 1):
                if j < len(neuron.weights):
                    original = neuron.weights[j]
                    neuron.weights[j] = original + h; plus = J()       # noqa: E702
                    neuron.weights[j] = original - h; minus = J()      # noqa: E702
                    neuron.weights[j] = original                       # WICHTIG: zurücksetzen
                else:
                    original = neuron.bias
                    neuron.bias = original + h; plus = J()             # noqa: E702
                    neuron.bias = original - h; minus = J()            # noqa: E702
                    neuron.bias = original
                g.append((plus - minus) / (2 * h))
            ergebnis.append(np.array(g))
    return ergebnis


def one_hot(k, n):
    y = np.zeros(n)
    y[k] = 1.0
    return y


@pytest.mark.parametrize("sizes, aktivierung, loss, ziel", [
    # die Kombination aus train_digits.py, verkleinert
    ([6, 5, 4], ["sigmoid", "softmax"], "cross_entropy", "one_hot"),
    # Softmax mit MSE: hier läuft die volle Jacobi-Matrix in Layer.backward
    ([6, 5, 4], ["tanh", "softmax"], "mse", "one_hot"),
    # Sigmoid-Ausgang mit Cross-Entropy, Ziele beliebig in [0, 1]
    ([6, 5, 3], ["relu", "sigmoid"], "cross_entropy", "uniform"),
    # zwei Hidden-Schichten vor der Softmax
    ([6, 7, 5, 4], ["sigmoid", "tanh", "softmax"], "cross_entropy", "one_hot"),
])
def test_gradienten_mit_softmax_und_cross_entropy(sizes, aktivierung, loss, ziel):
    np.random.seed(7)
    net = NeuralNetwork(sizes, activation=aktivierung, init="xavier", loss=loss)
    x = np.random.uniform(-1, 1, sizes[0])
    y = one_hot(2, sizes[-1]) if ziel == "one_hot" else np.random.uniform(0, 1, sizes[-1])

    analytisch = analytische_gradienten(net, x, y)
    numerisch = numerische_gradienten(net, x, y)

    abweichung = max(np.max(np.abs(a - n)) for a, n in zip(analytisch, numerisch))
    groesse = max(np.max(np.abs(n)) for n in numerisch)
    assert groesse > 1e-4, "Gradienten sind praktisch 0 -- Testaufbau prüfen"
    assert abweichung < 1e-6, f"Backprop weicht um {abweichung:.2e} ab"


def test_cross_entropy_kosten():
    net = NeuralNetwork([2, 3], activation="softmax", loss="cross_entropy")
    # J = -log(a_richtig)
    assert net.cost(np.array([0.2, 0.5, 0.3]), one_hot(1, 3)) == pytest.approx(-np.log(0.5))
    # a = 0 darf nicht zu log(0) = -inf führen
    assert np.isfinite(net.cost(np.array([0.0, 1.0, 0.0]), one_hot(0, 3)))


def test_kleines_klassifikationsproblem_wird_gelernt():
    """Drei Punktwolken in 2D, 3 Klassen: nach kurzem Training alle Trainingspunkte richtig."""
    rng = np.random.default_rng(3)
    mitten = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
    X = np.vstack([m + rng.normal(0, 0.4, (20, 2)) for m in mitten])
    Y = np.repeat(np.eye(3), 20, axis=0)
    np.random.seed(3)
    net = NeuralNetwork([2, 6, 3], activation=["tanh", "softmax"], init="xavier",
                        loss="cross_entropy")
    history = net.train(X, Y, epochs=60, learning_rate=0.1, shuffle=True)
    assert history[-1] < history[0] / 5
    treffer = sum(np.argmax(net.predict(x)) == np.argmax(y) for x, y in zip(X, Y))
    assert treffer == len(X)


# ===========================================================================
# 3. Export
# ===========================================================================

def test_loss_wird_gespeichert_und_geladen(tmp_path):
    np.random.seed(1)
    net = NeuralNetwork([4, 5, 3], activation=["sigmoid", "softmax"], loss="cross_entropy")
    pfad = tmp_path / "netz.json"
    net.dump_json(pfad)
    geladen = NeuralNetwork.load(pfad)
    assert geladen.loss == "cross_entropy"
    x = [0.1, 0.9, 0.4, 0.0]
    assert np.array_equal(geladen.predict(x), net.predict(x))


def test_alte_exporte_bleiben_unveraendert():
    """Netze mit MSE schreiben kein "loss"-Feld -- models/xor.json usw. sehen aus wie vorher."""
    daten = NeuralNetwork([2, 2, 1]).to_dict()
    assert "loss" not in daten
    assert NeuralNetwork.from_dict(json.loads(json.dumps(daten))).loss == "mse"


def test_softmax_laesst_sich_nicht_als_c_header_exportieren(tmp_path):
    net = NeuralNetwork([4, 3], activation="softmax")
    with pytest.raises(ValueError, match="gibt es im C-Code nicht"):
        net.dump_initfile(tmp_path / "netz.h")


# ===========================================================================
# 4. Falsche Kombinationen
# ===========================================================================

def test_unpassende_kostenfunktion_wird_abgelehnt():
    with pytest.raises(ValueError, match="Unbekannte Kostenfunktion"):
        NeuralNetwork([2, 2], loss="hinge")
    with pytest.raises(ValueError, match="softmax oder sigmoid"):
        NeuralNetwork([2, 2], activation="tanh", loss="cross_entropy")
