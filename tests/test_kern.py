"""
Tests für den Kern des Netzes (Phase 1).

Was hier nachgewiesen wird:
    1. Die Ableitungen der Aktivierungsfunktionen stimmen.
    2. Die Backpropagation berechnet die richtigen Gradienten -- geprüft gegen einen
       numerischen Gradienten, der von der Backpropagation völlig unabhängig ist.
    3. Das umgebaute Paket rechnet bit-identisch wie der Ausgangsstand.
    4. XOR wird gelernt.
    5. to_dict()/from_dict() erzeugen ein identisches Netz (Grundlage für den JSON-Export).
    6. Die Datensätze stimmen mit der Angabe überein.
    7. Falsche Eingaben werden mit verständlichen Fehlermeldungen abgefangen.

Ausführen (im Projektordner):
    python -m pytest -v
"""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from datasets import SEGMENT_NAMES, SEVEN_SEG_X, SEVEN_SEG_Y, XOR_X, XOR_Y
from nn import Layer, NeuralNetwork, Neuron, activations
from nn.neuron import weight_std

PROJEKT = Path(__file__).resolve().parent.parent


# ===========================================================================
# 1. Aktivierungsfunktionen
# ===========================================================================

@pytest.mark.parametrize("name", sorted(activations.ACTIVATIONS))
def test_ableitung_stimmt_mit_numerischer_ableitung_ueberein(name):
    """f'(z) aus dem Code gegen (f(z+h) - f(z-h)) / 2h, für jede registrierte Aktivierung."""
    f, df = activations.get(name)
    # Stellen bewusst nicht bei 0 -- dort hat ReLU einen Knick und keine Ableitung
    z = np.array([-3.0, -1.2, -0.3, 0.4, 1.1, 2.7])
    h = 1e-6
    numerisch = (f(z + h) - f(z - h)) / (2 * h)
    analytisch = df(z, f(z))   # Signatur derivative(z, a) mit a = f(z)
    assert np.max(np.abs(analytisch - numerisch)) < 1e-8


def test_unbekannte_aktivierung_gibt_verstaendlichen_fehler():
    with pytest.raises(ValueError, match="Bekannt sind"):
        activations.get("swish")
    with pytest.raises(ValueError):
        NeuralNetwork([2, 2, 1], activation="sigmoidd")   # Tippfehler


def test_weight_std():
    """Streuung der Startgewichte laut Docstring von weight_std()."""
    assert weight_std("klein", 2, 2) == 0.1
    assert weight_std("xavier", 2, 2) == pytest.approx(np.sqrt(0.5))       # XOR hidden
    assert weight_std("xavier", 4, 16) == pytest.approx(np.sqrt(2 / 20))   # 7-Seg hidden
    with pytest.raises(ValueError):
        weight_std("gross", 2, 2)


# ===========================================================================
# 2. Gradient-Check: Backpropagation gegen numerischen Gradienten
# ===========================================================================

def kosten(net, x, y):
    """J = 1/2 · Summe (A - Y)²  -- dieselbe Kostenfunktion wie in train()."""
    output = net.forward(x)
    return 0.5 * np.sum((output - np.array(y, dtype=float)) ** 2)


def analytische_gradienten(net, x, y):
    """
    Die Gradienten, wie sie die Backpropagation berechnet.

    Trick: backward() wendet die Gradienten sofort an (W_neu = W_alt - alpha · dJ/dW).
    Mit alpha = 1 auf einer KOPIE des Netzes gilt also  dJ/dW = W_alt - W_neu.
    So wird genau der Code geprüft, der auch beim Training läuft -- ohne ihn für den
    Test umzubauen.

    Rückgabe: Liste mit einem Eintrag (dJ/dw als Vektor, dJ/db als Zahl) pro Neuron.
    """
    kopie = copy.deepcopy(net)
    output = kopie.forward(x)
    kopie.backward(output, y, learning_rate=1.0)
    gradienten = []
    for alt_layer, neu_layer in zip(net.layers, kopie.layers):
        for alt, neu in zip(alt_layer.neurons, neu_layer.neurons):
            gradienten.append((alt.weights - neu.weights, alt.bias - neu.bias))
    return gradienten


def numerische_gradienten(net, x, y, h=1e-6):
    """
    Gradienten per zentraler Differenz:  dJ/dw ≈ (J(w + h) - J(w - h)) / 2h

    Benutzt nur Forward-Passes und weiß nichts von Kettenregel oder Deltas -- ist
    also eine völlig unabhängige Kontrolle. Jedes Gewicht wird einzeln kurz verschoben
    und danach wieder exakt auf den Originalwert zurückgesetzt.

    Rückgabe: gleicher Aufbau wie analytische_gradienten().
    """
    gradienten = []
    for layer in net.layers:
        for neuron in layer.neurons:
            g_w = np.zeros_like(neuron.weights)
            for j in range(len(neuron.weights)):
                original = neuron.weights[j]
                neuron.weights[j] = original + h
                plus = kosten(net, x, y)
                neuron.weights[j] = original - h
                minus = kosten(net, x, y)
                neuron.weights[j] = original   # WICHTIG: zurücksetzen
                g_w[j] = (plus - minus) / (2 * h)

            original = neuron.bias
            neuron.bias = original + h
            plus = kosten(net, x, y)
            neuron.bias = original - h
            minus = kosten(net, x, y)
            neuron.bias = original
            gradienten.append((g_w, (plus - minus) / (2 * h)))
    return gradienten


@pytest.mark.parametrize("sizes, aktivierung", [
    ([3, 5, 2], "sigmoid"),
    ([3, 5, 2], "tanh"),
    ([3, 5, 2], ["relu", "sigmoid"]),
    # 3 Schichten: prüft auch, dass der Fehler über ZWEI Hidden-Schichten richtig zurückläuft
    ([3, 5, 4, 2], ["tanh", "relu", "sigmoid"]),
    # die Architektur des 7-Segment-Decoders aus der Angabe
    ([4, 16, 7], "sigmoid"),
])
def test_backpropagation_stimmt_mit_numerischem_gradienten_ueberein(sizes, aktivierung):
    np.random.seed(42)
    # "xavier" statt Standard "klein": größere Gewichte -> deutlich von 0 verschiedene
    # Gradienten auch in den Hidden-Schichten, der Test wird dadurch schärfer
    net = NeuralNetwork(sizes, activation=aktivierung, init="xavier")
    x = np.random.uniform(-1, 1, sizes[0])
    y = np.random.uniform(0, 1, sizes[-1])

    analytisch = analytische_gradienten(net, x, y)
    numerisch = numerische_gradienten(net, x, y)

    max_abweichung = 0.0
    max_gradient = 0.0
    for (a_w, a_b), (n_w, n_b) in zip(analytisch, numerisch):
        max_abweichung = max(max_abweichung, np.max(np.abs(a_w - n_w)), abs(a_b - n_b))
        max_gradient = max(max_gradient, np.max(np.abs(n_w)), abs(n_b))

    # Sonst wäre der Test sinnlos: 0 == 0 stimmt immer
    assert max_gradient > 1e-4, "Gradienten sind praktisch 0 -- Testaufbau prüfen"
    assert max_abweichung < 1e-6, f"Backprop weicht um {max_abweichung:.2e} ab"


# ===========================================================================
# 3. Gleichwertigkeit mit dem Ausgangsstand
# ===========================================================================

def lade_ausgangsstand():
    """Lädt archiv/neural_network_ausgangsstand.py als Modul (liegt nicht im Suchpfad)."""
    pfad = PROJEKT / "archiv" / "neural_network_ausgangsstand.py"
    spec = importlib.util.spec_from_file_location("ausgangsstand", pfad)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)   # führt nur Definitionen aus, das Training steht unter __main__
    return modul


def alle_gewichte(net):
    """Alle Gewichte und Biases eines Netzes als eine flache Liste von Arrays."""
    return [np.append(n.weights, n.bias) for layer in net.layers for n in layer.neurons]


def test_rechnet_bit_identisch_wie_der_ausgangsstand():
    """
    Mit init="klein" und gleichem Seed muss das neue Paket EXAKT dieselben Zahlen liefern
    wie der alte Code -- über alle 20000 Epochen. Das beweist, dass der Umbau die
    Mathematik nicht verändert hat (auch nicht durch die neue Ableitung über a statt z).
    """
    alt = lade_ausgangsstand()

    np.random.seed(1)
    netz_alt = alt.NeuralNetwork([2, 2, 1], learning_rate=0.1)
    np.random.seed(1)
    netz_neu = NeuralNetwork([2, 2, 1], learning_rate=0.1, init="klein")

    for g_alt, g_neu in zip(alle_gewichte(netz_alt), alle_gewichte(netz_neu)):
        assert np.array_equal(g_alt, g_neu), "Startgewichte unterscheiden sich"

    verlauf_alt = netz_alt.train(XOR_X, XOR_Y, epochs=20000)
    verlauf_neu = netz_neu.train(XOR_X, XOR_Y, epochs=20000)

    # == statt approx: bit-identisch, nicht nur "ungefähr gleich"
    assert verlauf_alt == verlauf_neu
    for g_alt, g_neu in zip(alle_gewichte(netz_alt), alle_gewichte(netz_neu)):
        assert np.array_equal(g_alt, g_neu)
    # und die Werte, die der Ausgangsstand beim Ausführen ausgegeben hat
    assert [round(float(netz_neu.predict(x)[0]), 6) for x in XOR_X] == \
        [0.037759, 0.958389, 0.958241, 0.052827]


# ===========================================================================
# 4. Lernen
# ===========================================================================

def test_standard_initialisierung_ist_klein():
    """Entscheidung E-11 in CLAUDE.md: Standard ist "klein", also gleiche Startgewichte
    wie im Ausgangsstand, wenn kein init angegeben wird."""
    np.random.seed(1)
    standard = NeuralNetwork([2, 2, 1])
    np.random.seed(1)
    klein = NeuralNetwork([2, 2, 1], init="klein")
    for a, b in zip(alle_gewichte(standard), alle_gewichte(klein)):
        assert np.array_equal(a, b)


def test_xor_wird_mit_den_parametern_der_angabe_gelernt():
    """2-2-1, 20000 Epochen, alpha 0.1, Seed 1 -> alle 4 Fälle richtig."""
    np.random.seed(1)
    net = NeuralNetwork([2, 2, 1], learning_rate=0.1)
    net.train(XOR_X, XOR_Y, epochs=20000)
    for x, y in zip(XOR_X, XOR_Y):
        assert int(net.predict(x)[0] > 0.5) == y[0]


def test_shuffle_ist_mit_seed_reproduzierbar():
    verlaeufe = []
    for _ in range(2):
        np.random.seed(7)
        verlaeufe.append(NeuralNetwork([2, 2, 1]).train(XOR_X, XOR_Y, epochs=200, shuffle=True))
    assert verlaeufe[0] == verlaeufe[1]

    # und es wird wirklich gemischt: anderer Verlauf als in fester Reihenfolge
    np.random.seed(7)
    ohne = NeuralNetwork([2, 2, 1]).train(XOR_X, XOR_Y, epochs=200)
    assert ohne != verlaeufe[0]


def test_erste_schicht_rechnet_keinen_fehler_mehr_zurueck():
    """compute_prev_errors=False spart nur Rechenzeit -- die Gewichte ändern sich gleich."""
    np.random.seed(0)
    mit = Layer(3, 2)
    ohne = copy.deepcopy(mit)
    fehler = np.array([0.1, -0.2, 0.3])
    for layer in (mit, ohne):
        layer.forward([0.5, -0.5])

    assert mit.backward(fehler, 0.1, compute_prev_errors=True) is not None
    assert ohne.backward(fehler, 0.1, compute_prev_errors=False) is None
    for a, b in zip(mit.neurons, ohne.neurons):
        assert np.array_equal(a.weights, b.weights) and a.bias == b.bias


# ===========================================================================
# 5. to_dict / from_dict
# ===========================================================================

def test_to_dict_und_from_dict_ergeben_dasselbe_netz():
    np.random.seed(3)
    net = NeuralNetwork([4, 16, 7], activation=["tanh", "sigmoid"])
    # kurz trainieren, damit nicht nur Startwerte verglichen werden
    net.train(SEVEN_SEG_X, SEVEN_SEG_Y, epochs=50)

    # einmal durch JSON schicken -> beweist, dass das Dictionary JSON-tauglich ist
    daten = json.loads(json.dumps(net.to_dict()))

    zustand_vorher = np.random.get_state()
    kopie = NeuralNetwork.from_dict(daten)
    zustand_nachher = np.random.get_state()
    # Laden darf keine Zufallszahlen verbrauchen
    assert np.array_equal(zustand_vorher[1], zustand_nachher[1])
    assert zustand_vorher[2] == zustand_nachher[2]

    assert kopie.layer_sizes == [4, 16, 7]
    assert [layer.activation_name for layer in kopie.layers] == ["tanh", "sigmoid"]
    for x in SEVEN_SEG_X:
        # bit-identisch: JSON speichert floats verlustfrei
        assert np.array_equal(net.predict(x), kopie.predict(x))


def test_from_dict_erkennt_unpassende_daten():
    np.random.seed(0)
    daten = NeuralNetwork([2, 3, 1]).to_dict()
    daten["layer_sizes"] = [2, 4, 1]                              # Struktur passt nicht
    with pytest.raises(ValueError):
        NeuralNetwork.from_dict(daten)

    daten = NeuralNetwork([2, 3, 1]).to_dict()
    daten["layers"][1]["neurons"][0]["weights"].append(0.5)      # ein Gewicht zu viel
    with pytest.raises(ValueError):
        NeuralNetwork.from_dict(daten)


# ===========================================================================
# 6. Datensätze
# ===========================================================================

def test_xor_tabelle():
    for x, y in zip(XOR_X, XOR_Y):
        assert y[0] == x[0] ^ x[1]   # ^ ist XOR in Python


def test_siebensegment_eingaben_sind_die_binaerdarstellung_der_ziffer():
    assert len(SEVEN_SEG_X) == len(SEVEN_SEG_Y) == 10
    for ziffer, bits in enumerate(SEVEN_SEG_X):
        # 8-4-2-1, höchstwertiges Bit zuerst
        assert bits == [(ziffer >> 3) & 1, (ziffer >> 2) & 1, (ziffer >> 1) & 1, ziffer & 1]


def test_siebensegment_ausgaben():
    assert SEGMENT_NAMES == list("abcdefg")
    for y in SEVEN_SEG_Y:
        assert len(y) == 7 and set(y) <= {0, 1}
    # Stichproben, die man ohne Tabelle weiß
    assert SEVEN_SEG_Y[8] == [1, 1, 1, 1, 1, 1, 1]   # 8: alle Segmente an
    assert SEVEN_SEG_Y[1] == [0, 1, 1, 0, 0, 0, 0]   # 1: nur rechts oben (b) und rechts unten (c)
    assert SEVEN_SEG_Y[0][6] == 0                    # 0: Mittelstrich (g) aus
    # alle 10 Muster verschieden -- sonst wären Ziffern nicht unterscheidbar
    assert len({tuple(y) for y in SEVEN_SEG_Y}) == 10


# ===========================================================================
# 7. Eingabeprüfung
# ===========================================================================

@pytest.mark.parametrize("inputs, targets", [
    ([[0, 0], [1, 1]], [[0]]),   # unterschiedlich viele Eingaben und Ziele
    ([[0, 0, 1]], [[0]]),        # Eingabe zu lang
    ([[0, 0]], [[0, 1]]),        # Ziel zu lang
    ([], []),                    # gar keine Daten
])
def test_train_prueft_die_trainingsdaten(inputs, targets):
    with pytest.raises(ValueError):
        NeuralNetwork([2, 2, 1]).train(inputs, targets, epochs=1)


@pytest.mark.parametrize("sizes, aktivierung", [
    ([2], "sigmoid"),                  # nur Eingang, keine Schicht
    ([2, 0, 1], "sigmoid"),            # Schicht ohne Neuronen
    ([2, 2, 1], ["sigmoid"]),          # 2 Schichten, aber nur 1 Aktivierung
])
def test_konstruktor_prueft_die_struktur(sizes, aktivierung):
    with pytest.raises(ValueError):
        NeuralNetwork(sizes, activation=aktivierung)


def test_neuron_prueft_die_anzahl_der_gewichte():
    with pytest.raises(ValueError):
        Neuron(3, weights=[0.1, 0.2])
