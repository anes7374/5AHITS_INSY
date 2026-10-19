"""
Trainingsdaten aus der Angabe.

Die Tabellen sind wörtlich aus 4AHITS_NN_from_scratch.pdf übernommen, damit man sie Zeile
für Zeile gegen das PDF prüfen kann:
    - XOR:        Part 2, Step 1
    - 7-Segment:  Part 2, Step 3 (Seite 10)

Die Tests in tests/test_kern.py prüfen zusätzlich, dass die 7-Segment-Eingaben wirklich
die Binärdarstellung der Ziffer sind und die Tabelle nicht versehentlich verschoben ist.
"""

# ---------------------------------------------------------------------------
# XOR (Part 2, Step 1)
# ---------------------------------------------------------------------------
# Ausgabe 1, wenn genau einer der beiden Eingänge 1 ist.
# Nicht linear trennbar: keine einzelne Gerade trennt die Einsen von den Nullen.
# Deshalb braucht das Netz eine Hidden-Schicht, um XOR lernen zu können.

XOR_X = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]

# Jeder Zielwert ist eine Liste, weil das Netz einen Ausgabe-Vektor liefert
# (hier mit nur einem Eintrag, da es 1 Output-Neuron gibt).
XOR_Y = [[0],
         [1],
         [1],
         [0]]


# ---------------------------------------------------------------------------
# 7-Segment-Decoder (Part 2, Step 3)
# ---------------------------------------------------------------------------
# Eingabe:  die Ziffer 0-9 als 4 Bit, höchstwertiges Bit zuerst (8-4-2-1).
#           Beispiel 6 = 0·8 + 1·4 + 1·2 + 0·1 -> [0, 1, 1, 0]
# Ausgabe:  7 Werte für die Segmente a bis g, 1 = Segment leuchtet.
#
# Lage der Segmente auf der Anzeige:
#
#        aaa
#       f   b
#       f   b
#        ggg
#       e   c
#       e   c
#        ddd

SEGMENT_NAMES = ["a", "b", "c", "d", "e", "f", "g"]

SEVEN_SEG_X = [[0, 0, 0, 0],   # 0
               [0, 0, 0, 1],   # 1
               [0, 0, 1, 0],   # 2
               [0, 0, 1, 1],   # 3
               [0, 1, 0, 0],   # 4
               [0, 1, 0, 1],   # 5
               [0, 1, 1, 0],   # 6
               [0, 1, 1, 1],   # 7
               [1, 0, 0, 0],   # 8
               [1, 0, 0, 1]]   # 9

#                a  b  c  d  e  f  g
SEVEN_SEG_Y = [[1, 1, 1, 1, 1, 1, 0],   # 0
               [0, 1, 1, 0, 0, 0, 0],   # 1
               [1, 1, 0, 1, 1, 0, 1],   # 2
               [1, 1, 1, 1, 0, 0, 1],   # 3
               [0, 1, 1, 0, 0, 1, 1],   # 4
               [1, 0, 1, 1, 0, 1, 1],   # 5
               [1, 0, 1, 1, 1, 1, 1],   # 6
               [1, 1, 1, 0, 0, 0, 0],   # 7
               [1, 1, 1, 1, 1, 1, 1],   # 8
               [1, 1, 1, 1, 0, 1, 1]]   # 9
