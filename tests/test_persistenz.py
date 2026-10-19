"""
Tests für Persistenz (Phase 2), trainierte Modelle (Phase 3) und die Trainer-Optionen
Mini-Batch/Momentum (Phase 8).

Was hier nachgewiesen wird:
    1. JSON, TXT und Header speichern das Netz exakt (bit-identische Gewichte und Vorhersagen),
       auch mit tanh/ReLU und mehreren Hidden-Schichten.
    2. Beschädigte Dateien werden mit verständlichen Fehlern abgelehnt.
    3. Die exportierten Modelle in models/ erfüllen die Angabe (XOR 4/4, 7-Segment 70/70)
       und liefern genau die Vorhersagen, die direkt nach dem Training protokolliert wurden.
    4. Mini-Batch und Momentum rechnen richtig: Full-Batch = Mittelwert der Einzelgradienten,
       Batch 1 ohne Momentum = SGD, Momentum beschleunigt XOR.

Ausführen (im Projektordner):
    python -m pytest -v
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from auswertung import MODELS, genauigkeit, lade_protokoll, vorhersagen
from datasets import SEVEN_SEG_X, SEVEN_SEG_Y, XOR_X, XOR_Y
from nn import NeuralNetwork, persistence

# Netze, an denen der Export geprüft wird: die beiden aus der Angabe plus Sonderfälle
NETZE = [
    ([2, 2, 1], "sigmoid"),
    ([4, 16, 7], "sigmoid"),
    ([3, 5, 4, 2], ["tanh", "relu", "sigmoid"]),
]


def zufallsnetz(sizes, aktivierung, seed=11):
    """Ein kurz trainiertes Netz, damit die Gewichte "krumme" Zahlen sind."""
    np.random.seed(seed)
    net = NeuralNetwork(sizes, activation=aktivierung, init="xavier")
    x = np.random.rand(5, sizes[0]).tolist()
    y = np.random.rand(5, sizes[-1]).round().tolist()
    net.train(x, y, epochs=20)
    return net


def gleiche_gewichte(a, b):
    """True, wenn zwei Netze bit-identische Gewichte, Biases und Aktivierungen haben."""
    if a.layer_sizes != b.layer_sizes:
        return False
    for la, lb in zip(a.layers, b.layers):
        if la.activation_name != lb.activation_name:
            return False
        for na, nb in zip(la.neurons, lb.neurons):
            if not (np.array_equal(na.weights, nb.weights) and na.bias == nb.bias):
                return False
    return True


# ===========================================================================
# 1. Exakter Roundtrip
# ===========================================================================

@pytest.mark.parametrize("sizes, aktivierung", NETZE)
def test_json_roundtrip_ist_bit_exakt(tmp_path, sizes, aktivierung):
    net = zufallsnetz(sizes, aktivierung)
    pfad = net.dump_json(tmp_path / "netz.json")
    geladen = NeuralNetwork.load(pfad)
    assert gleiche_gewichte(net, geladen)
    assert geladen.learning_rate == net.learning_rate
    x = np.random.rand(sizes[0])
    assert np.array_equal(net.predict(x), geladen.predict(x))


@pytest.mark.parametrize("sizes, aktivierung", NETZE)
def test_txt_roundtrip_ist_bit_exakt(tmp_path, sizes, aktivierung):
    net = zufallsnetz(sizes, aktivierung)
    pfad = net.dump_txt(tmp_path / "netz.txt", "Test")
    assert gleiche_gewichte(net, NeuralNetwork.load_txt(pfad))


def test_txt_ist_menschenlesbar_und_ascii(tmp_path):
    """Part 1, Step 5: Anzahl Layer, Neuronen, Gewichte und Biases stehen erklärt drin."""
    net = zufallsnetz([4, 16, 7], "sigmoid")
    text = net.dump_txt(tmp_path / "netz.txt", "7-Segment").read_bytes().decode("ascii")
    assert "# Struktur: 4 -> 16 -> 7" in text
    assert "16 4 sigmoid" in text and "7 16 sigmoid" in text
    assert text.count("# Neuron ") == 16 + 7
    # Datenzeilen sind rein numerisch (der C-Parser kennt keine Kommentare mitten in der Zeile)
    for zeile in text.splitlines():
        if zeile.strip() and not zeile.lstrip().startswith("#") and "sigmoid" not in zeile:
            [float(v) for v in zeile.split()]


def header_werte(text, name):
    """Liest die Zahlen eines Arrays NN_<name> aus einem generierten Header."""
    block = re.search(rf"NN_{name}\[[^=]*=\s*\{{(.*?)\}};", text, re.S).group(1)
    block = re.sub(r"/\*.*?\*/", "", block)   # Kommentare entfernen
    return [float(v) for v in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", block)]


@pytest.mark.parametrize("sizes, aktivierung", NETZE)
def test_header_enthaelt_alle_werte_exakt(tmp_path, sizes, aktivierung):
    """Part 1, Step 6: Größen, Biases und Gewichte stehen exakt im Header."""
    net = zufallsnetz(sizes, aktivierung)
    text = net.dump_initfile(tmp_path / "netz_net.h", "Test").read_text(encoding="utf-8")
    assert f"#define NN_NUM_LAYERS {len(sizes) - 1}" in text
    assert "nn_init_from_header(Network *net)" in text
    assert [int(v) for v in header_werte(text, "LAYER_SIZES")] == sizes
    for i, layer in enumerate(net.layers):
        assert header_werte(text, f"L{i}_BIASES") == [n.bias for n in layer.neurons]
        assert header_werte(text, f"L{i}_WEIGHTS") == [w for n in layer.neurons for w in n.weights]


def test_laden_verbraucht_keine_zufallszahlen(tmp_path):
    net = zufallsnetz([2, 2, 1], "sigmoid")
    net.dump_json(tmp_path / "a.json")
    net.dump_txt(tmp_path / "a.txt")
    np.random.seed(5)
    erwartet = np.random.rand()
    np.random.seed(5)
    NeuralNetwork.load(tmp_path / "a.json")
    NeuralNetwork.load_txt(tmp_path / "a.txt")
    assert np.random.rand() == erwartet


# ===========================================================================
# 2. Beschädigte Dateien
# ===========================================================================

def test_kaputte_json_wird_erkannt(tmp_path):
    data = zufallsnetz([2, 2, 1], "sigmoid").to_dict()
    data["layers"][1]["neurons"][0]["weights"].append(0.5)   # 3 statt 2 Gewichte
    pfad = tmp_path / "kaputt.json"
    pfad.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        NeuralNetwork.load(pfad)


@pytest.mark.parametrize("ersetzung, meldung", [
    (lambda t: t.rsplit("\n", 3)[0] + "\n", "endet zu früh"),
    (lambda t: t.replace("2 2 sigmoid", "2 2 swish"), "Unbekannte Aktivierung"),
    (lambda t: t.replace("1 2 sigmoid", "1 3 sigmoid"), "erwartet Bias"),
])
def test_kaputte_txt_wird_erkannt(tmp_path, ersetzung, meldung):
    pfad = zufallsnetz([2, 2, 1], "sigmoid").dump_txt(tmp_path / "netz.txt")
    pfad.write_text(ersetzung(pfad.read_text()))
    with pytest.raises(ValueError, match=meldung):
        NeuralNetwork.load_txt(pfad)


# ===========================================================================
# 3. Die trainierten Modelle aus Part 2
# ===========================================================================

braucht_modelle = pytest.mark.skipif(
    not (MODELS / "7seg.json").exists() or not (MODELS / "xor.json").exists(),
    reason="erst train_xor.py und train_7seg.py ausführen",
)


@braucht_modelle
@pytest.mark.parametrize("name, X, Y, bits", [
    ("xor", XOR_X, XOR_Y, 4),
    ("7seg", SEVEN_SEG_X, SEVEN_SEG_Y, 70),
])
def test_exportierte_modelle_erfuellen_die_angabe(name, X, Y, bits):
    referenz = np.array(lade_protokoll(MODELS / f"{name}_training.json")["vorhersagen"])
    for net in (NeuralNetwork.load(MODELS / f"{name}.json"), NeuralNetwork.load_txt(MODELS / f"{name}.txt")):
        ausgaben = vorhersagen(net, X)
        assert genauigkeit(ausgaben, Y)[0] == bits
        assert np.array_equal(ausgaben, referenz)     # Export/Import ohne jede Abweichung


@braucht_modelle
def test_header_passt_zum_json_modell():
    net = NeuralNetwork.load(MODELS / "7seg.json")
    text = (MODELS / "7seg_net.h").read_text(encoding="utf-8")
    for i, layer in enumerate(net.layers):
        assert header_werte(text, f"L{i}_WEIGHTS") == [w for n in layer.neurons for w in n.weights]


# ===========================================================================
# 4. Mini-Batch und Momentum
# ===========================================================================

def test_full_batch_ist_der_gemittelte_gradient():
    """
    Ein Full-Batch-Schritt muss genau dem Mittelwert der Einzelgradienten entsprechen.
    Die Einzelgradienten werden wie in test_kern.py gewonnen: backward() mit alpha = 1 auf
    einer Kopie ergibt dJ/dW = W_alt - W_neu.
    """
    np.random.seed(3)
    net = NeuralNetwork([4, 5, 7], init="xavier")
    erwartet = [np.zeros((len(L.neurons), L.num_inputs + 1)) for L in net.layers]
    for x, y in zip(SEVEN_SEG_X, SEVEN_SEG_Y):
        kopie = NeuralNetwork.from_dict(net.to_dict())
        vorher = [np.array([np.append(n.weights, n.bias) for n in L.neurons]) for L in kopie.layers]
        kopie.backward(kopie.forward(x), y, 1.0)
        nachher = [np.array([np.append(n.weights, n.bias) for n in L.neurons]) for L in kopie.layers]
        for g, a, b in zip(erwartet, vorher, nachher):
            g += (a - b) / len(SEVEN_SEG_X)

    alt = [np.array([np.append(n.weights, n.bias) for n in L.neurons]) for L in net.layers]
    net.train(SEVEN_SEG_X, SEVEN_SEG_Y, epochs=1, learning_rate=0.5, batch_size=len(SEVEN_SEG_X))
    neu = [np.array([np.append(n.weights, n.bias) for n in L.neurons]) for L in net.layers]
    for g, a, b in zip(erwartet, alt, neu):
        np.testing.assert_allclose(a - b, 0.5 * g, rtol=0, atol=1e-14)


def test_batch_1_ohne_momentum_entspricht_sgd():
    """Der Batch-Weg mit Batchgröße 1 rechnet dasselbe wie SGD (bis auf Rundung)."""
    np.random.seed(4)
    a = NeuralNetwork([2, 3, 1])
    b = NeuralNetwork.from_dict(a.to_dict())
    order = list(range(4))
    for _ in range(200):
        a._sgd_epoch(XOR_X, XOR_Y, order, 0.5)
        b._batch_epoch(XOR_X, XOR_Y, order, 0.5, 1, 0.0)
    np.testing.assert_allclose(a.predict([0, 1]), b.predict([0, 1]), rtol=0, atol=1e-12)


def test_momentum_beschleunigt_xor():
    """Mit Momentum 0.9 erreicht XOR die Kosten 0.01 deutlich früher als reines SGD."""
    def epochen_bis(momentum):
        np.random.seed(1)
        verlauf = NeuralNetwork([2, 2, 1]).train(XOR_X, XOR_Y, epochs=20000, momentum=momentum)
        return next((i for i, k in enumerate(verlauf) if k < 0.01), None)

    ohne, mit = epochen_bis(0.0), epochen_bis(0.9)
    assert ohne is not None and mit is not None
    assert mit < ohne / 3


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"momentum": 1.0}, {"momentum": -0.1}])
def test_ungueltige_traineroptionen(kwargs):
    with pytest.raises(ValueError):
        NeuralNetwork([2, 2, 1]).train(XOR_X, XOR_Y, epochs=1, **kwargs)
