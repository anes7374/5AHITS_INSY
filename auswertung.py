"""
Gemeinsame Hilfsfunktionen zum Auswerten und Anzeigen der Ergebnisse.

Wird von den Skripten zu Part 2 (train_xor.py, test_xor_load.py, train_7seg.py,
test_7seg_load.py) und von den Experimenten benutzt, damit Tabellen, Schwellwert und
Genauigkeitsmaße überall gleich berechnet werden.

Begriffe:
    Schwellwert   Eine Sigmoid-Ausgabe > 0.5 zählt als 1, sonst als 0
                  (Optionale Erweiterung auf Seite 11: "thresholding").
    Bit-Accuracy  Anteil richtiger Einzel-Ausgaben, beim 7-Segment-Netz also von 70 Bits.
    Ziffern-Acc.  Anteil der Ziffern, bei denen ALLE 7 Segmente stimmen -- das strengere
                  Maß, denn ein einziges falsches Segment macht die Anzeige falsch.
"""

import json
from pathlib import Path

import numpy as np

from datasets import SEGMENT_NAMES

PROJEKT = Path(__file__).resolve().parent
MODELS = PROJEKT / "models"

SCHWELLWERT = 0.5


def binaer(ausgabe):
    """Netzausgabe -> Liste aus 0/1 per Schwellwert 0.5."""
    return [int(wert > SCHWELLWERT) for wert in ausgabe]


def vorhersagen(net, inputs):
    """
    Alle Vorhersagen eines Netzes als 2D-numpy-Array.

    Rückgabe: Form (Anzahl Samples, Anzahl Ausgänge)
    """
    return np.array([net.predict(x) for x in inputs])


def genauigkeit(ausgaben, targets):
    """
    Bit- und Sample-Genauigkeit nach Schwellwert.

    ausgaben: Netzausgaben, Form (Samples, Ausgänge)
    targets:  Sollwerte (0/1), gleiche Form
    Rückgabe: (richtige Bits, alle Bits, richtige Samples, alle Samples)
              -- ein Sample gilt nur dann als richtig, wenn alle seine Bits stimmen.
    """
    ist = np.array(ausgaben) > SCHWELLWERT
    soll = np.array(targets) > SCHWELLWERT
    bits_ok = ist == soll
    return (int(bits_ok.sum()), int(bits_ok.size),
            int(bits_ok.all(axis=1).sum()), int(len(soll)))


def mse(ausgaben, targets):
    """Kosten wie im Training: Mittel über die Samples von 1/2 · Summe (A - Y)²."""
    diff = np.array(ausgaben) - np.array(targets)
    return float(np.mean(0.5 * np.sum(diff ** 2, axis=1)))


# ---------------------------------------------------------------------------
# 7-Segment-Anzeige im Terminal
# ---------------------------------------------------------------------------

def siebensegment_ascii(bits):
    """
    Zeichnet eine Ziffer als 3 Zeilen ASCII-Art, zur Sichtkontrolle im Terminal.

    bits:     7 Werte (0/1) für die Segmente a-g
    Rückgabe: Liste mit 3 Zeichenketten zu je 3 Zeichen

        _      Zeile 0:  a
       |_|     Zeile 1:  f g b
       |_|     Zeile 2:  e d c
    """
    a, b, c, d, e, f, g = [bool(x) for x in bits]
    return [
        " " + ("_" if a else " ") + " ",
        ("|" if f else " ") + ("_" if g else " ") + ("|" if b else " "),
        ("|" if e else " ") + ("_" if d else " ") + ("|" if c else " "),
    ]


def ziffern_nebeneinander(liste_von_bits):
    """Mehrere ASCII-Ziffern nebeneinander, mit zwei Leerzeichen Abstand."""
    zeichen = [siebensegment_ascii(bits) for bits in liste_von_bits]
    return "\n".join("  ".join(z[zeile] for z in zeichen) for zeile in range(3))


def segment_string(bits):
    """Sichtbare Segmente als Buchstaben, z.B. [1,1,0,1,1,0,1] -> 'ab de g'."""
    return "".join(name if bit else " " for name, bit in zip(SEGMENT_NAMES, bits))


# ---------------------------------------------------------------------------
# Trainingsprotokoll (für Tests, Plots und die Visualisierung)
# ---------------------------------------------------------------------------

def speichere_protokoll(pfad, **daten):
    """
    Speichert Metadaten eines Trainingslaufs als JSON (z.B. Seed, Dauer, Kostenverlauf
    und die Vorhersagen direkt nach dem Training).

    Die Vorhersagen dienen als REFERENZ: Die Test-Skripte, der C-Code und die
    HTML-Visualisierung müssen genau diese Werte reproduzieren.
    numpy-Arrays werden in Listen umgewandelt, floats bleiben exakt (json nutzt repr).
    """
    def umwandeln(wert):
        if isinstance(wert, np.ndarray):
            return wert.tolist()
        if isinstance(wert, (np.floating, np.integer)):
            return wert.item()
        raise TypeError(f"Nicht JSON-tauglich: {type(wert)}")

    pfad = Path(pfad)
    pfad.parent.mkdir(parents=True, exist_ok=True)
    with open(pfad, "w", encoding="utf-8") as f:
        json.dump(daten, f, default=umwandeln)
    return pfad


def lade_protokoll(pfad):
    """Gegenstück zu speichere_protokoll()."""
    with open(pfad, encoding="utf-8") as f:
        return json.load(f)
