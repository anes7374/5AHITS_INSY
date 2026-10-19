# Neuronales Netz from scratch

Abgabe zu **4AHITS_NN_from_scratch.pdf** (INSY, 4AHITS). Ein Feedforward-Netz mit
Backpropagation, selbst implementiert in Python, exportiert in drei Formate, nachgerechnet in
C und TensorFlow, und als interaktive Visualisierung im Browser.

| Dokument | Inhalt |
|---|---|
| [Implementation.md](Implementation.md) | Plan mit allen Phasen und ihrem Stand |
| [CLAUDE.md](CLAUDE.md) | Entscheidungsprotokoll: warum etwas so gebaut ist, mit Messwerten |
| [Ausgangsstand.md](Ausgangsstand.md) | Bestandsaufnahme des Codes vor Projektbeginn |
| [Prompts.md](Prompts.md) | gekürztes Protokoll der Zusammenarbeit |
| [dokumentation/dokumentation.pdf](dokumentation/dokumentation.pdf) | **Abgabe-Dokumentation** (Quarto, auch als HTML) mit Theorie, Formeln, Code, Screenshots und Ergebnissen |

---

## Ergebnisse auf einen Blick

| Teil der Angabe | Ergebnis |
|---|---|
| Part 1 · Neuron, Layer, NeuralNetwork, Backprop | Gradient-Check gegen numerische Ableitung: Abweichung 7·10⁻¹¹ |
| Part 1 · JSON, TXT, Header | alle drei Formate speichern die Gewichte bit-exakt |
| Part 2 · XOR 2-2-1 | 4/4 richtig, Kosten 9,6·10⁻⁴, Abweichung nach dem Laden: 0 |
| Part 2 · 7-Segment 4-16-7 | 70/70 Segmente, 10/10 Ziffern, Abweichung nach dem Laden: 0 |
| Part 3 · TensorFlow | mit gleichen Startgewichten identisch bis 10⁻¹⁶; Vergleichstabelle unten |
| Part 4 · C, Init aus `.txt` | 70/70, Abweichung zu Python 2·10⁻¹⁶ |
| Part 5 · C, Init aus `.h` | 70/70, Abweichung zu Python 2·10⁻¹⁶ |
| Part 6 · ersetzt durch Visualisierung | `visualisierung/index.html`, rechnet im Browser bit-identisch zu C |
| Bonus · ReLU/tanh, Mini-Batch, Momentum, Tests | alle umgesetzt, 70 Tests grün |
| Phase 9 · Ziffern zeichnen (Softmax, Cross-Entropy) | 96,7 % auf 359 Testbildern, live im Browser |

---

## Schnellstart

Voraussetzung: Python 3.14 mit `numpy` (für Tests `pytest`, für Grafiken `matplotlib`).
Aufruf über den Launcher `py -3.14`: Seit der Installation von Python 3.12 für TensorFlow
startet `python` dieses 3.12, und dort fehlen numpy und pytest.

```
py -3.14 train_xor.py          # Part 2, Step 1 -> models/xor.json/.txt/_net.h
py -3.14 test_xor_load.py      # Part 2, Step 2 -> Abweichung 0, 4/4
py -3.14 train_7seg.py         # Part 2, Step 3 -> models/7seg.json/.txt/_net.h   (~30 s)
py -3.14 test_7seg_load.py     # Part 2, Step 4 -> 70/70 Bits, 10/10 Ziffern
py -3.14 train_digits.py       # Phase 9 -> models/digits.json, 97,7 % Test     (~5 min, braucht MNIST, siehe unten)
py -3.14 -m pytest -v          # 70 Tests
```

Danach `visualisierung/index.html` per Doppelklick im Browser öffnen.

### Visualisierung neu erzeugen

```
py -3.14 visualisierung/export_daten.py
```

Schreibt die aktuellen Gewichte aus `models/` plus alle Vergleichswerte (C, TensorFlow,
Experimente) in `visualisierung/index.html`. Die Vorlage ist `visualisierung/vorlage.html`.
Was fehlt (z.B. C nicht gebaut), wird übersprungen und auf der Seite als fehlend markiert.

Die Seite hat fünf Bereiche (Tabs oben), es ist immer nur ein Kapitel sichtbar:
**Decoder** (Ziffer wählen, Netz beobachten, alle 16 Eingaben) · **Zeichnen** (Phase 9) ·
**So rechnet es** (Schritt für Schritt, ein Neuron, alle Gewichte, alle Eingaben, Lernen) ·
**Training** (Verlauf und TensorFlow, Live-Training, XOR) · **Ergebnisse** (Kennzahlen,
Nachweis der Gleichheit aller Implementierungen, Experimente). Kapitel lassen sich direkt
verlinken, z.B. `index.html#neuron`.

### Dokumentation neu erzeugen

Benötigt [Quarto](https://quarto.org) (PDF über das mitgelieferte Typst, kein LaTeX nötig).

```
py -3.14 dokumentation/screenshots.py      # Screenshots der Seite (Edge, helles Schema)
py -3.14 dokumentation/abbildungen.py      # Sigmoid, MNIST-Diagramme, Kopien aus ergebnisse/
quarto render dokumentation/dokumentation.qmd   # -> dokumentation.html und dokumentation.pdf
```

### C (Part 4 und 5)

Benötigt `gcc` und `make`. Unter Windows: `winget install BrechtSanders.WinLibs.POSIX.UCRT`
(MinGW-w64; `make` heißt dort `mingw32-make`).

```
cd c_forward      && mingw32-make && nn_txt ..\models\7seg.txt     # Part 4
cd c_header_init  && mingw32-make && nn_header                     # Part 5
```

`--roh` gibt nur die Zahlen aus; so vergleicht `tests/test_c.py` C und Python automatisch.

### TensorFlow (Part 3)

TensorFlow läuft nicht unter Python 3.14, daher ein eigenes venv mit Python 3.12:

```
py -3.12 -m venv .venv-tf
.venv-tf\Scripts\python -m pip install tensorflow
.venv-tf\Scripts\python compare_tensorflow.py     # ~2 Minuten -> ergebnisse/tf_vergleich.json
```

### Experimente (Phase 8)

```
py -3.14 experimente/seed_studie.py     # XOR: lokale Minima nach Netzgröße und Init   (~2 min)
py -3.14 experimente/aktivierungen.py   # Sigmoid / tanh / ReLU                       (~3 min)
py -3.14 experimente/rauschen.py        # Rausch-Robustheit, gekippte Bits           (~3 min)
py -3.14 experimente/vektorisiert.py    # Benchmark Objekte gegen Matrizen            (~1 min)
py -3.14 experimente/plots.py           # Loss-Kurven als PNG
```

Alle Ergebnisse landen als JSON und PNG in `ergebnisse/`.

---

## Projektstruktur

```
nn/                     der Kern ("from scratch", numpy nur für Vektorrechnung)
  activations.py          sigmoid, tanh, relu + Ableitungen, Auswahl über den Namen
  neuron.py               Neuron: a = f(w·x + b), Update, Mini-Batch/Momentum
  layer.py                Layer: Forward, Backward (Kettenregel), Export
  network.py              NeuralNetwork: train, predict, backward, dump_json/load/...
  persistence.py          JSON, TXT (auch für C), C-Header
datasets.py             XOR- und 7-Segment-Tabelle aus der Angabe
auswertung.py           Schwellwert, Genauigkeit, ASCII-Anzeige, Trainingsprotokoll
train_xor.py ... test_7seg_load.py     Part 2
train_digits.py         Phase 9: Ziffernerkennung 196-64-10 auf MNIST (Softmax, Cross-Entropy)
compare_tensorflow.py   Part 3 (im venv .venv-tf)
c_forward/              Part 4: nn_forward.h/.c, test_7seg.c, main_txt.c, Makefile
c_header_init/          Part 5: main_header.c, Makefile (nutzt c_forward/ und models/7seg_net.h)
models/                 exportierte Netze und Trainingsprotokolle
visualisierung/         Part-6-Ersatz: vorlage.html, export_daten.py -> index.html
experimente/            Phase 8: parallel.py (viele Netze gleichzeitig), Studien, Plots
ergebnisse/             JSON und PNG aus TensorFlow-Vergleich und Experimenten
tests/                  pytest: Kern, Persistenz, Trainer-Optionen, C, Softmax
dokumentation/          Quarto-Dokumentation: dokumentation.qmd -> .html und .pdf, Screenshots, Abbildungen
archiv/                 Ausgangsstand, unverändert
```

---

## Wichtige Messergebnisse

### Part 3: eigene Implementierung gegen TensorFlow 2.21

20 000 Epochen, α = 0,1, SGD mit einem Sample pro Schritt, Sigmoid.

| Netz | Lauf | Kosten J | richtig | J < 0,01 ab Epoche | Dauer |
|---|---|---|---|---|---|
| 2-2-1 | eigen | 9,6·10⁻⁴ | 4/4 | 12 494 | 2,6 s |
| | TF identisch | 9,6·10⁻⁴ | 4/4 | 12 494 | 11,5 s |
| | TF Standard | 8,4·10⁻² | **3/4** | nie | 11,5 s |
| 4-16-7 | eigen | 4,4·10⁻⁴ | 70/70 | 2 626 | 33,2 s |
| | TF identisch | 4,4·10⁻⁴ | 70/70 | 2 626 | 17,9 s |
| | TF Standard | 2,5·10⁻³ | 70/70 | 9 166 | 19,0 s |
| 4-10-7 | eigen | 6,1·10⁻⁴ | 70/70 | 2 991 | 26,2 s |
| | TF identisch | 6,1·10⁻⁴ | 70/70 | 2 991 | 16,7 s |
| | TF Standard | 2,7·10⁻³ | 70/70 | 9 243 | 18,8 s |

*TF identisch* = gleiche Startgewichte, gleiche Kostenfunktion ½·Σ(A−Y)², float64.
*TF Standard* = Keras-Voreinstellungen (Glorot-Init, Keras-MSE, float32). Die Keras-MSE teilt
durch die Anzahl der Ausgänge und trainiert dadurch effektiv mit kleinerer Lernrate; bei XOR
landet sie in einem lokalen Minimum.

### Phase 8

- **Seed-Studie XOR** (100 Seeds): Mit 2 Hidden-Neuronen hängen 7 (klein) bzw. 24 (Xavier)
  Läufe dauerhaft fest. Ab 3 Hidden-Neuronen: 0 – und Xavier ist dann ~3× schneller.
- **Aktivierungen**: 7-Segment lernt mit tanh/ReLU 4–6× schneller als mit Sigmoid. XOR 2-2-1
  scheitert mit tanh/ReLU meistens (zu wenig Reserve, „tote“ ReLU-Neuronen).
- **Rauschen**: Bei σ = 0,2 auf den Eingängen erkennt das Modell 78 % der Ziffern komplett
  richtig, ein mit Rauschen trainiertes Netz 96 %.
- **Vektorisiert**: 3,8× schneller bei 4-16-7, 9,7× bei 4-64-7, aber langsamer bei XOR 2-2-1.
- **Momentum / Mini-Batch** (XOR, Seed 1): SGD erreicht J < 0,01 nach 12 494 Epochen,
  mit Momentum 0,9 nach 1 310 (fast 10× schneller), mit 0,5 nach 6 191. Full-Batch (4 Samples
  pro Update) ohne Momentum kommt in 20 000 Epochen nicht vom Plateau J ≈ 0,125 weg, weil es
  nur ein Viertel der Updates macht; mit Momentum 0,9 schafft es Full-Batch nach 11 941.

### Phase 9: Ziffern zeichnen

- **Netz** 196 → 64 (Sigmoid) → 10 (Softmax), Cross-Entropy, trainiert mit `nn/` auf MNIST
  (60 000 Trainings-, 10 000 Testbilder, auf 14×14 verkleinert): **97,7 % richtig** auf den
  Testbildern.
- **Daten holen** (einmalig, 11 MB, landet in `~/.keras/datasets/mnist.npz`):
  `.venv-tf\Scripts\python -c "from tensorflow import keras; keras.datasets.mnist.load_data()"`
- **Softmax** ist eine Aktivierung der ganzen Schicht: Die Neuronen liefern z, der Layer normiert.
  Gradient-Check für Softmax und Cross-Entropy < 10⁻⁶ (`tests/test_softmax.py`).
- **Im Browser** wird die Zeichnung genau wie MNIST aufbereitet: längere Seite auf 20 Pixel,
  Schwerpunkt in die Mitte von 28×28, dann auf 14×14 verkleinert. Größe und Lage der Zeichnung
  spielen deshalb keine Rolle.
