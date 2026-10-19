"""
Aktivierungsfunktionen und ihre Ableitungen.

Bezug zur Angabe: Part 1, Step 2 ("Use sigmoid (or configurable) activation function")
und Bonus-Aufgabe "Add support for different activation functions (ReLU, tanh)".

Warum es dieses Modul gibt
--------------------------
Im Ausgangsstand hat jedes Neuron seine Aktivierung als Funktionsobjekt gespeichert.
Ein Funktionsobjekt lässt sich aber nicht in eine JSON-Datei schreiben (Part 1, Step 4).
Deshalb hat hier jede Aktivierung einen **Namen** ("sigmoid", "tanh", "relu"), und die
Registry ACTIVATIONS übersetzt diesen Namen in die passenden Funktionen. In der JSON-Datei
steht später nur der Name, beim Laden holt get(name) die Funktionen wieder.

Einheitliche Signatur der Ableitungen: derivative(z, a)
-------------------------------------------------------
    z = gewichtete Summe VOR der Aktivierung   (in der Angabe: Z)
    a = Ausgabe NACH der Aktivierung, a = f(z) (in der Angabe: A)

Beide Werte hat das Neuron aus dem Forward-Pass ohnehin gespeichert. Jede Ableitung nimmt
sich den Wert, mit dem sie am günstigsten zu berechnen ist:
    sigmoid:  f'(z) = a · (1 - a)    -> braucht nur a  (genau die Formel aus der Angabe)
    tanh:     f'(z) = 1 - a²         -> braucht nur a
    relu:     f'(z) = 1 wenn z > 0   -> braucht z
So muss keine Ableitung die Aktivierung ein zweites Mal ausrechnen
(Schwachstelle 3 in Ausgangsstand.md).

Alle Funktionen arbeiten mit einzelnen Zahlen und mit numpy-Arrays (elementweise).
Ausnahme: softmax (Phase 9) wirkt auf eine ganze Schicht, siehe LAYER_WISE.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Sigmoid
# ---------------------------------------------------------------------------

def sigmoid(z):
    """
    sigma(z) = 1 / (1 + e^(-z))  -- quetscht jede Zahl in den Bereich (0, 1).

    Große positive z -> fast 1, große negative z -> fast 0, z = 0 -> genau 0.5.
    Standard-Aktivierung dieses Projekts, weil die Angabe sie verwendet und weil die
    Ausgabe (0 bis 1) direkt als "Bit ist an / aus" gelesen werden kann.
    """
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z, a):
    """
    Ableitung der Sigmoid-Funktion:  sigma'(z) = sigma(z) · (1 - sigma(z)) = a · (1 - a)

    In der Angabe:  dA/dZ = A · (1 - A).
    z wird nicht gebraucht, steht aber wegen der einheitlichen Signatur in der Parameterliste.
    Rückgabe: maximal 0.25 (bei a = 0.5), gegen 0, wenn a nahe 0 oder 1 liegt.
    """
    return a * (1.0 - a)


# ---------------------------------------------------------------------------
# tanh
# ---------------------------------------------------------------------------

def tanh(z):
    """
    Tangens hyperbolicus -- wie Sigmoid s-förmig, aber im Bereich (-1, 1) und um 0 zentriert.

    Weil die Ausgaben um 0 liegen statt um 0.5, lernen Hidden-Schichten damit oft schneller.
    Für die Output-Schicht bei 0/1-Zielen (XOR, 7-Segment) passt Sigmoid besser.
    """
    return np.tanh(z)


def tanh_derivative(z, a):
    """
    Ableitung von tanh:  tanh'(z) = 1 - tanh(z)² = 1 - a²

    Maximal 1 (bei a = 0), gegen 0, wenn a nahe -1 oder 1 liegt.
    """
    return 1.0 - a * a


# ---------------------------------------------------------------------------
# ReLU
# ---------------------------------------------------------------------------

def relu(z):
    """
    ReLU (Rectified Linear Unit):  relu(z) = max(0, z)

    Negative Werte werden zu 0, positive bleiben unverändert. Sehr billig zu berechnen
    und in tiefen Netzen Standard. Nachteil: Ein Neuron, dessen z immer negativ ist,
    gibt immer 0 aus, bekommt Gradient 0 und lernt nie wieder ("totes Neuron").
    """
    return np.maximum(0.0, z)


def relu_derivative(z, a):
    """
    Ableitung von ReLU:  1 für z > 0, sonst 0.

    Bei genau z = 0 ist ReLU nicht differenzierbar (Knick). Wir nehmen dort 0 --
    das ist die übliche Festlegung und spielt praktisch keine Rolle, weil z fast nie
    exakt 0 ist.
    Hier wird z statt a gebraucht: aus a = 0 allein ließe sich nicht unterscheiden,
    ob z negativ oder genau 0 war.

    (z > 0) ergibt True/False, mal 1.0 wird daraus 1.0/0.0 -- funktioniert für
    einzelne Zahlen und für Arrays gleichermaßen.
    """
    return (z > 0) * 1.0


# ---------------------------------------------------------------------------
# Linear (Identität)
# ---------------------------------------------------------------------------

def linear(z):
    """
    Keine Aktivierung:  a = z.

    Wird gebraucht als Neuron-Anteil der Softmax-Schicht (siehe unten): Dort gibt jedes
    Neuron nur seine Summe z weiter, und erst der Layer rechnet daraus die Softmax.
    """
    return z


def linear_derivative(z, a):
    """Ableitung der Identität: immer 1 (als Array, wenn z ein Array ist)."""
    return np.ones_like(z, dtype=float)


# ---------------------------------------------------------------------------
# Softmax -- Aktivierung für eine GANZE Schicht (Phase 9, Ziffernerkennung)
# ---------------------------------------------------------------------------
# Sigmoid, tanh und ReLU wirken auf jedes Neuron für sich. Softmax dagegen macht aus den
# Summen z ALLER Neuronen einer Schicht eine Wahrscheinlichkeitsverteilung:
#
#     a_i = e^(z_i) / Summe_j e^(z_j)        alle a_i > 0, zusammen genau 1
#
# Damit passt sie für "welche von 10 Ziffern ist es?": Die Ausgabe i ist die Wahrscheinlichkeit
# für Ziffer i. Weil ein einzelnes Neuron die anderen z nicht kennt, ist die Arbeit geteilt:
#   - jedes Neuron rechnet nur z = w·x + b und gibt es unverändert weiter ("linear")
#   - Layer.forward() wendet softmax() auf den ganzen Vektor an
#   - Layer.backward() benutzt die Ableitung der ganzen Schicht (Jacobi-Matrix, siehe dort)

def softmax(z):
    """
    Softmax über einen Vektor z (Länge = Neuronen der Schicht).

    Rückgabe: Vektor gleicher Länge, Einträge in (0, 1), Summe 1.

    Vor dem e^ wird max(z) abgezogen. Das ändert das Ergebnis nicht (Zähler und Nenner werden
    mit demselben Faktor e^(-max) multipliziert), verhindert aber einen Überlauf: e^800 wäre
    unendlich, e^0 = 1 ist das Größte, was hier vorkommt.
    """
    e = np.exp(z - np.max(z))
    return e / np.sum(e)


# ---------------------------------------------------------------------------
# Registry: Name -> (Funktion, Ableitung)
# ---------------------------------------------------------------------------

# WICHTIG: Die Namen hier sind das, was in den exportierten Dateien (.json, .txt) steht.
# Einen Namen umbenennen heißt, dass alte Exportdateien nicht mehr geladen werden können.
#
# WICHTIG: Der Eintrag "softmax" beschreibt nur, was das EINZELNE Neuron einer Softmax-Schicht
# tut -- nämlich z unverändert weitergeben. Die eigentliche Softmax rechnet der Layer
# (LAYER_WISE unten). So bleibt der Name für Neuron, Layer und Export derselbe.
ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
    "relu": (relu, relu_derivative),
    "linear": (linear, linear_derivative),
    "softmax": (linear, linear_derivative),
}

# Aktivierungen, die der Layer über die ganze Schicht anwendet: Name -> Funktion(Vektor z)
LAYER_WISE = {
    "softmax": softmax,
}


def get(name):
    """
    Liefert zu einem Namen die Aktivierungsfunktion und ihre Ableitung.

    name:     z.B. "sigmoid"
    Rückgabe: Tupel (funktion, ableitung)
    Fehler:   ValueError mit der Liste der bekannten Namen, wenn der Name unbekannt ist --
              z.B. bei einem Tippfehler oder einer beschädigten Exportdatei.
    """
    try:
        return ACTIVATIONS[name]
    except KeyError:
        bekannt = ", ".join(sorted(ACTIVATIONS))
        raise ValueError(
            f"Unbekannte Aktivierungsfunktion '{name}'. Bekannt sind: {bekannt}"
        ) from None
