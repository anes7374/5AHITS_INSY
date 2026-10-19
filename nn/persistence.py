"""
Speichern und Laden eines Netzes in drei Formaten.

Bezug zur Angabe: Part 1, Step 4-6
    Step 4  dump_json() / load()     JSON: vollständiges Netz, verlustfrei wieder ladbar
    Step 5  dump_txt()               TXT:  menschenlesbar UND vom C-Code (Part 4) parsebar
            load_txt()               (zusätzlich) Python-Gegenstück zum C-Parser
    Step 6  dump_initfile()          .h:   C-Header, mit dem Part 5 das Netz ohne Datei
                                           initialisiert

Benutzung:
    from nn import persistence
    persistence.dump_json(net, "models/xor.json")
    net2 = persistence.load("models/xor.json")
oder direkt über die Methoden am Netz (net.dump_json(...), NeuralNetwork.load(...)),
die nur hierher weiterleiten.

Verlustfreiheit -- das zentrale Ziel dieser Datei
-------------------------------------------------
Alle drei Formate schreiben die Gewichte mit repr(float). repr liefert die KÜRZESTE
Dezimaldarstellung, aus der beim Einlesen wieder exakt dieselbe double-Zahl entsteht
(z.B. 0.1 statt 0.1000000000000000055511151231257827). Das gilt für Pythons float(),
für C's strtod()/sscanf("%lf") und für C-Quelltext-Literale gleichermaßen. Ergebnis:
Python, C (aus .txt) und C (aus .h) rechnen mit bit-identischen Gewichten.
Ein Format wie "%.6f" wäre lesbarer, würde aber pro Gewicht bis zu 5e-7 Fehler einbauen.
"""

import json
from pathlib import Path

from .layer import Layer
from .neuron import Neuron

# Kennzahlen der Aktivierungen im C-Code (enum NnActivation in c_forward/nn_forward.h).
# WICHTIG: muss mit dem enum in nn_forward.h übereinstimmen, sonst rechnet das C-Programm
# mit der falschen Funktion.
C_ACTIVATION_CONSTANTS = {
    "sigmoid": "NN_ACT_SIGMOID",
    "tanh": "NN_ACT_TANH",
    "relu": "NN_ACT_RELU",
}


def _num(value):
    """Eine Zahl als kürzeste exakte Dezimaldarstellung (siehe Dateikopf: Verlustfreiheit)."""
    return repr(float(value))


def _network_class():
    """
    Liefert die Klasse NeuralNetwork.

    Der Import steht in einer Funktion statt oben in der Datei, weil network.py seinerseits
    diese Datei importiert (für net.dump_json() usw.). Ein Import auf Modulebene in beide
    Richtungen wäre ein Zirkelimport.
    """
    from .network import NeuralNetwork
    return NeuralNetwork


# ===========================================================================
# Step 4: JSON
# ===========================================================================

def dump_json(net, path):
    """
    Speichert das Netz als JSON-Datei.

    net:  das NeuralNetwork
    path: Zieldatei, z.B. "models/xor.json". Fehlende Ordner werden angelegt.

    Inhalt: genau net.to_dict() -- layer_sizes, learning_rate und pro Schicht die
    Aktivierung und für jedes Neuron Bias und Gewichte. Der json-Modul schreibt floats
    mit repr, der Roundtrip ist also bit-exakt.
    Rückgabe: der Pfad als Path-Objekt
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(net.to_dict(), f, indent=2)
        f.write("\n")
    return path


def load(path):
    """
    Lädt ein Netz aus einer JSON-Datei von dump_json().

    Rückgabe: ein neues NeuralNetwork mit genau den gespeicherten Gewichten.
              Verbraucht keine Zufallszahlen.
    Fehler:   ValueError, wenn die Datei nicht zusammenpassende Größen enthält
              (die Prüfung macht NeuralNetwork.from_dict).
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return _network_class().from_dict(data)


# ===========================================================================
# Step 5: TXT -- menschenlesbar und vom C-Code parsebar
# ===========================================================================
#
# Format (eine Regel macht beides möglich: alles nach '#' am Zeilenanfang ist Kommentar):
#
#   # Kommentarzeilen erklären dem Menschen, was die Zahlen bedeuten
#   2                         <- Anzahl Schichten (ohne Eingabe)
#   16 4 sigmoid              <- Schicht: Neuronen, Eingänge je Neuron, Aktivierung
#   b w0 w1 w2 w3             <- eine Zeile pro Neuron: Bias, danach die Gewichte
#   ...
#   7 16 sigmoid
#   b w0 ... w15
#
# Der C-Parser (c_forward/nn_forward.c) liest zeilenweise, überspringt Leer- und
# Kommentarzeilen und liest die Zahlen der Reihe nach. load_txt() unten macht in Python
# genau dasselbe -- damit lässt sich prüfen, dass Datei und Parser zusammenpassen.

def dump_txt(net, path, title="Neuronales Netz"):
    """
    Speichert das Netz als kommentierte Textdatei (Format siehe oben).

    net:   das NeuralNetwork
    path:  Zieldatei, z.B. "models/7seg.txt"
    title: Überschrift in der ersten Kommentarzeile

    Pro Neuron steht eine Zeile "Bias Gewicht0 Gewicht1 ...", darüber ein Kommentar mit
    der Nummer des Neurons. Die Zahlen sind in Spalten ausgerichtet, damit man sie leicht
    vergleichen kann (z.B. "welches Hidden-Neuron reagiert stark auf Bit 0?").

    Der Text ist bewusst ASCII (ae statt ä): Der C-Code und jeder Editor lesen ihn dann
    ohne Rücksicht auf die Zeichenkodierung.
    Rückgabe: der Pfad als Path-Objekt
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    struktur = " -> ".join(str(size) for size in net.layer_sizes)
    num_layers = len(net.layers)

    lines = [
        f"# {title}",
        f"# Struktur: {struktur}   (Eingabe -> ... -> Ausgabe)",
        f"# Aktivierungen: {', '.join(layer.activation_name for layer in net.layers)}",
        "#",
        "# Aufbau dieser Datei -- Zeilen mit '#' sind Kommentare und werden beim Einlesen",
        "# uebersprungen, alle anderen Zeilen enthalten nur Zahlen bzw. den Aktivierungsnamen:",
        "#   1. Anzahl der Schichten (ohne Eingabeschicht)",
        "#   2. pro Schicht eine Kopfzeile:  <Neuronen> <Eingaenge je Neuron> <Aktivierung>",
        "#   3. danach pro Neuron eine Zeile: <Bias> <Gewicht 0> <Gewicht 1> ...",
        "#      Gewicht j gehoert zu Eingang j, also zu Neuron j der Vorschicht.",
        "#",
        "# Rechnung je Neuron:  z = Bias + Summe(Gewicht_j * Eingang_j),  Ausgabe = f(z)",
        "# Die Zahlen sind exakt (kuerzeste Darstellung, die beim Einlesen wieder genau",
        "# denselben double-Wert ergibt).",
        "",
        "# Anzahl der Schichten",
        str(num_layers),
    ]

    for index, layer in enumerate(net.layers):
        rolle = "Ausgabe" if index == num_layers - 1 else "Hidden"
        lines += [
            "",
            f"# ---- Schicht {index} ({rolle}): {layer.num_neurons} Neuronen, "
            f"je {layer.num_inputs} Eingaenge, {layer.activation_name}",
            f"{layer.num_neurons} {layer.num_inputs} {layer.activation_name}",
            "#   " + "".join(
                f"{name:>24}" for name in
                ["Bias"] + [f"Gewicht {j}" for j in range(layer.num_inputs)]
            ),
        ]
        for n_index, neuron in enumerate(layer.neurons):
            werte = [neuron.bias] + list(neuron.weights)
            # Die Neuronennummer steht in einer eigenen Kommentarzeile davor, nicht hinter
            # den Zahlen -- so bleibt jede Datenzeile rein numerisch und der C-Parser muss
            # keine Kommentare mitten in einer Zeile erkennen.
            lines.append(f"# Neuron {n_index}")
            lines.append("    " + "".join(f"{_num(w):>24}" for w in werte))

    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def _data_lines(path):
    """
    Liefert die Datenzeilen einer TXT-Datei: ohne Kommentar- und Leerzeilen,
    jeweils schon in Wörter zerlegt. Gleiche Regel wie im C-Parser.
    """
    with open(path, encoding="ascii") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                yield stripped.split()


def load_txt(path, learning_rate=0.1):
    """
    Lädt ein Netz aus einer TXT-Datei von dump_txt().

    Gegenstück zum C-Parser: Wenn Python aus der .txt dasselbe Netz bekommt wie aus der
    .json, dann ist die .txt vollständig und exakt -- und der C-Parser liest dieselben
    Zeilen nach derselben Regel.

    learning_rate: steht nicht in der TXT-Datei (C braucht sie nicht, dort wird nicht
                   trainiert); wird nur gesetzt, falls man das Netz weitertrainieren will.
    Rückgabe: ein neues NeuralNetwork
    Fehler:   ValueError bei fehlenden Zeilen, falscher Zahl an Werten oder unbekannter
              Aktivierung.
    """
    rows = _data_lines(path)

    def next_row(was):
        try:
            return next(rows)
        except StopIteration:
            raise ValueError(f"{path}: Datei endet zu früh, erwartet: {was}") from None

    num_layers = int(next_row("Anzahl der Schichten")[0])
    layers = []
    for index in range(num_layers):
        kopf = next_row(f"Kopfzeile von Schicht {index}")
        if len(kopf) != 3:
            raise ValueError(f"{path}: Kopfzeile von Schicht {index} braucht 3 Werte: {kopf}")
        num_neurons, num_inputs, activation = int(kopf[0]), int(kopf[1]), kopf[2]
        neurons = []
        for n_index in range(num_neurons):
            werte = [float(v) for v in next_row(f"Schicht {index}, Neuron {n_index}")]
            if len(werte) != num_inputs + 1:
                raise ValueError(
                    f"{path}: Schicht {index}, Neuron {n_index}: erwartet Bias + "
                    f"{num_inputs} Gewichte, gefunden {len(werte)} Werte"
                )
            neurons.append(Neuron(num_inputs, activation, weights=werte[1:], bias=werte[0]))
        layers.append(Layer(num_neurons, num_inputs, activation, neurons=neurons))

    # Die Struktur ergibt sich aus den Schichten: Eingänge der ersten, dann alle Ausgänge
    layer_sizes = [layers[0].num_inputs] + [layer.num_neurons for layer in layers]
    return _network_class()(layer_sizes, learning_rate, layers=layers)


# ===========================================================================
# Step 6: C-Header
# ===========================================================================

def dump_initfile(net, path, title="Neuronales Netz"):
    """
    Erzeugt einen C-Header, mit dem ein C-Programm das Netz ohne Datei-Parsing aufbaut.

    net:   das NeuralNetwork
    path:  Zieldatei, z.B. "models/7seg_net.h"
    title: Beschreibung im Kopfkommentar

    Inhalt des Headers:
        NN_NUM_LAYERS                    Anzahl der Schichten (ohne Eingabe)
        NN_LAYER_SIZES[]                 z.B. {4, 16, 7}
        NN_ACTIVATIONS[]                 Aktivierung pro Schicht (enum aus nn_forward.h)
        NN_L<i>_BIASES[n]                Biases der Schicht i
        NN_L<i>_WEIGHTS[n][m]            Gewichte der Schicht i, Zeile = Neuron
        nn_init_from_header(Network*)    baut damit ein Network auf

    Der Header inkludiert nn_forward.h aus Part 4 und nutzt dessen Structs und
    nn_alloc() -- Part 5 verwendet also dieselbe Datenstruktur und dieselbe nn_forward()
    wie Part 4, nur die Quelle der Zahlen ist eine andere.

    Alles im Header ist 'static': Würde er in zwei .c-Dateien eingebunden, gäbe es sonst
    doppelte Definitionen beim Linken.
    Rückgabe: der Pfad als Path-Objekt
    Fehler:   ValueError, wenn eine Aktivierung im C-Code nicht existiert (z.B. softmax) --
              sonst entstünde ein Header, der nicht kompiliert.
    """
    for layer in net.layers:
        if layer.activation_name not in C_ACTIVATION_CONSTANTS:
            raise ValueError(
                f"Aktivierung '{layer.activation_name}' gibt es im C-Code nicht. "
                f"Möglich sind: {', '.join(C_ACTIVATION_CONSTANTS)}"
            )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Include-Guard aus dem Dateinamen: 7seg_net.h -> NN_7SEG_NET_H
    guard = "NN_" + "".join(c if c.isalnum() else "_" for c in path.name.upper())
    struktur = " -> ".join(str(size) for size in net.layer_sizes)
    num_layers = len(net.layers)

    out = [
        "/*",
        f" * {path.name} -- {title}",
        f" * Struktur: {struktur}",
        " *",
        " * AUTOMATISCH ERZEUGT von nn/persistence.py (dump_initfile). Nicht von Hand",
        " * bearbeiten -- stattdessen das Netz neu trainieren und exportieren.",
        " *",
        " * Bezug zur Angabe: Part 1, Step 6 (Header erzeugen) und Part 5 (Initialisierung",
        " * aus dem Header). Benutzung:",
        " *",
        f" *     #include \"{path.name}\"",
        " *     Network net;",
        " *     if (nn_init_from_header(&net) != 0) { Fehler }",
        " *     ... nn_predict(&net, eingabe, ausgabe) ...",
        " *     nn_free(&net);",
        " *",
        " * Speicher: nn_init_from_header() reserviert über nn_alloc() Speicher und kopiert die",
        " * Konstanten hinein. Der Aufrufer gibt ihn mit nn_free() wieder frei -- genauso wie",
        " * bei einem Netz aus nn_load_txt(). Beide Wege liefern also gleich gebaute Netze.",
        " *",
        " * Die Zahlen sind exakt (kürzeste Dezimaldarstellung, die der C-Compiler wieder in",
        " * genau denselben double-Wert übersetzt wie Python).",
        " */",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        '#include "nn_forward.h"',
        "",
        "/* Anzahl der Schichten ohne Eingabeschicht */",
        f"#define NN_NUM_LAYERS {num_layers}",
        "",
        "/* Größen aller Schichten inkl. Eingabe: [0] = Eingänge, [i] = Neuronen von Schicht i-1 */",
        "static const int NN_LAYER_SIZES[NN_NUM_LAYERS + 1] = {"
        + ", ".join(str(s) for s in net.layer_sizes) + "};",
        "",
        "/* Aktivierungsfunktion pro Schicht */",
        "static const NnActivation NN_ACTIVATIONS[NN_NUM_LAYERS] = {"
        + ", ".join(C_ACTIVATION_CONSTANTS[layer.activation_name] for layer in net.layers)
        + "};",
    ]

    for index, layer in enumerate(net.layers):
        n, m = layer.num_neurons, layer.num_inputs
        rolle = "Ausgabe" if index == num_layers - 1 else "Hidden"
        out += [
            "",
            f"/* ---- Schicht {index} ({rolle}): {n} Neuronen, je {m} Eingänge, "
            f"{layer.activation_name} ---- */",
            f"static const double NN_L{index}_BIASES[{n}] = {{",
        ]
        out += [f"    {_num(neuron.bias)},  /* Neuron {k} */"
                for k, neuron in enumerate(layer.neurons)]
        out += [
            "};",
            f"/* Zeile k = Gewichte von Neuron k, Spalte j = Gewicht auf Eingang j */",
            f"static const double NN_L{index}_WEIGHTS[{n}][{m}] = {{",
        ]
        out += ["    {" + ", ".join(_num(w) for w in neuron.weights) + f"}},  /* Neuron {k} */"
                for k, neuron in enumerate(layer.neurons)]
        out.append("};")

    # Hilfsarrays, damit die Init-Funktion per Schleife über die Schichten laufen kann.
    # Die 2D-Arrays werden dafür als flache Zeiger auf ihr erstes Element abgelegt
    # (ein double[n][m] liegt im Speicher Zeile für Zeile hintereinander).
    out += [
        "",
        "/* Zeiger auf die Arrays je Schicht, für die Schleife in nn_init_from_header() */",
        "static const double *const NN_BIASES[NN_NUM_LAYERS] = {"
        + ", ".join(f"NN_L{i}_BIASES" for i in range(num_layers)) + "};",
        "static const double *const NN_WEIGHTS[NN_NUM_LAYERS] = {"
        + ", ".join(f"&NN_L{i}_WEIGHTS[0][0]" for i in range(num_layers)) + "};",
        "",
        "/*",
        " * Baut das Netz aus den Konstanten oben auf.",
        " *",
        " * net:      wird befüllt (vorheriger Inhalt wird überschrieben, nicht freigegeben)",
        " * Rückgabe: 0 bei Erfolg, -1 wenn nn_alloc() keinen Speicher bekommt (dann ist",
        " *           net leer und muss nicht freigegeben werden)",
        " * Speicher: gehört danach dem Aufrufer, freigeben mit nn_free(net).",
        " */",
        "static int nn_init_from_header(Network *net)",
        "{",
        "    if (nn_alloc(net, NN_NUM_LAYERS, NN_LAYER_SIZES, NN_ACTIVATIONS) != 0) {",
        "        return -1;",
        "    }",
        "    for (int l = 0; l < NN_NUM_LAYERS; l++) {",
        "        Layer *layer = &net->layers[l];",
        "        for (int k = 0; k < layer->num_neurons; k++) {",
        "            Neuron *neuron = &layer->neurons[k];",
        "            neuron->bias = NN_BIASES[l][k];",
        "            for (int j = 0; j < neuron->num_inputs; j++) {",
        "                /* Zeile k, Spalte j im flach abgelegten [n][m]-Array */",
        "                neuron->weights[j] = NN_WEIGHTS[l][k * neuron->num_inputs + j];",
        "            }",
        "        }",
        "    }",
        "    return 0;",
        "}",
        "",
        f"#endif /* {guard} */",
    ]

    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path
