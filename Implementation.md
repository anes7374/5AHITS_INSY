# Implementierungsplan

Stand: **19.10.2026** — Phasen 0–11 abgeschlossen · Ausgangslage siehe [Ausgangsstand.md](Ausgangsstand.md) ·
Entscheidungsprotokoll: [CLAUDE.md](CLAUDE.md)

---

## Leitgedanke

Zwei getrennte Ziele, in dieser Reihenfolge:

1. **Pflichtteil** — die Angabe `4AHITS_NN_from_scratch.pdf` vollständig abarbeiten
   (Phasen 0–7). Erst wenn das steht, geht es weiter.
2. **Kür** — gezielte Erweiterungen (Phase 8), die aus einer korrekten Arbeit eine
   exzellente machen.

**Planänderung gegenüber der Angabe:** Part 6 (ESP32 mit physischem 7-Segment-Display über
Serial) entfällt. An seiner Stelle steht eine **interaktive Visualisierung** des
7-Segment-Decoders, die den Rechenweg des Netzes Schritt für Schritt sichtbar macht.
Begründung siehe [CLAUDE.md](CLAUDE.md), Eintrag E-01.

---

## Zielstruktur des Projekts

```
Neuronales Netz/
├── CLAUDE.md                 # laufendes Entscheidungsprotokoll
├── Ausgangsstand.md          # Bestandsaufnahme
├── Implementation.md         # dieses Dokument
├── Prompts.md                # gekürztes Protokoll aller Prompts
├── README.md                 # Reproduktionsanleitung und Ergebnisse
├── neural_network.py         # bleibt als Kompatibilitäts-Einstieg
├── auswertung.py             # Schwellwert, Accuracy, ASCII-Anzeige, Trainingsprotokoll
├── archiv/                   # Ausgangsstand, unverändert (Ziel der Verweise in Ausgangsstand.md)
├── nn/                       # der Kern als Paket
│   ├── __init__.py
│   ├── activations.py        # sigmoid / relu / tanh + Ableitungen + Namens-Registry
│   ├── neuron.py
│   ├── layer.py
│   ├── network.py            # train / predict / backward
│   └── persistence.py        # dump_json, load, dump_txt, load_txt, dump_initfile
├── datasets.py               # XOR- und 7-Segment-Tabelle aus der Angabe
├── train_xor.py              # Part 2, Step 1
├── test_xor_load.py          # Part 2, Step 2
├── train_7seg.py             # Part 2, Step 3
├── test_7seg_load.py         # Part 2, Step 4
├── compare_tensorflow.py     # Part 3
├── models/                   # xor.json/.txt/_net.h, 7seg.json/.txt/_net.h, *_training.json
├── c_forward/                # Part 4: nn_forward.h/.c, test_7seg.c, main_txt.c, Makefile
├── c_header_init/            # Part 5: main_header.c + Header aus Part 1
├── tests/                    # pytest: test_kern.py seit Phase 1, Ausbau in Phase 8
├── pytest.ini                # Testkonfiguration
├── visualisierung/           # ESP32-Ersatz: vorlage.html, export_daten.py -> index.html
├── experimente/              # Phase 8: Studien, Benchmark, Plots
└── ergebnisse/               # JSON/PNG aus TF-Vergleich und Experimenten
```

Die Trainings- und Testskripte liegen bewusst im Projekt-Root, damit `import nn` ohne
Eingriffe in `sys.path` funktioniert.

---

## Phase 0 — Dokumentation aufsetzen ✅

- [x] `CLAUDE.md` mit den Grundsatzentscheidungen E-01 bis E-07
- [x] `Ausgangsstand.md` — Bestandsaufnahme inkl. Schwachstellenanalyse
- [x] `Implementation.md` — dieses Dokument
- [x] `Prompts.md` — gekürztes Protokoll aller Prompts (Regel R-02)
- [x] Regel R-01 (Kommentare) umgesetzt: Ausgangsstand vollständig kommentiert

`CLAUDE.md` wird in **jeder** folgenden Phase fortgeschrieben, nicht am Ende nachgetragen.

---

## Phase 1 — Part 1, Steps 1–3: Kern nach `nn/` überführen ✅

Die bestehende Logik wurde übernommen, nicht neu geschrieben. Ergänzt wurde nur, was für die
folgenden Parts fehlt. Entscheidungen: [CLAUDE.md](CLAUDE.md), E-08 bis E-13.

### `nn/activations.py`
- [x] `sigmoid`, `relu`, `tanh` mit ihren Ableitungen
- [x] Registry `ACTIVATIONS = {"sigmoid": (fn, dfn), ...}` plus `get(name)` —
  **Voraussetzung für die JSON-Persistenz**, behebt Schwachstelle 1
- [x] Ableitungen nutzen die **Ausgabe**: `A·(1−A)` statt erneutem Sigmoid. Das ist exakt
  die Formel aus der Angabe und spart die doppelten `exp`-Aufrufe (Schwachstelle 3).
  ReLU braucht weiterhin `z`, deshalb die einheitliche Signatur `derivative(z, a)`.

### `nn/neuron.py`, `nn/layer.py`, `nn/network.py`
- [x] `Neuron`: zusätzlich `activation_name`; Gewichte und Bias direkt übergebbar (zum Laden);
  `to_dict()` / `from_dict()`
- [x] `Layer`: `num_neurons`, `num_inputs`, `activation_name` als echte Felder
  (Schwachstelle 2); erzeugt die Startgewichte selbst; `forward` / `backward` aus dem Bestand
- [x] `NeuralNetwork`: merkt sich `layer_sizes`; Aktivierung pro Schicht wählbar; `backward`
  spart die Rückpropagierung in der Eingangsschicht (Schwachstelle 4); `train` prüft die
  Daten und kann optional mischen (`shuffle`)
- [x] `to_dict()` / `from_dict()` auf **allen drei** Ebenen (Plan: nur Neuron) → Phase 2 muss
  nur noch Dateien schreiben (E-12)
- [x] **Initialisierung — geändert gegenüber dem Plan.** Geplant war Xavier als Standard.
  Die Messung (E-11) ergab: Bei XOR 2-2-1 bleiben mit Xavier 24 von 100 Läufen dauerhaft
  hängen, mit dem bisherigen `0.1` nur 7 von 100; beim 7-Segment-Netz gibt es keinen
  Unterschied. **Standard bleibt `init="klein"`**, `init="xavier"` ist als Option da —
  implementiert mit der echten Glorot-Formel `sqrt(2/(n_in + n_out))`, nicht der im Plan
  genannten LeCun-Formel (E-10).

### `datasets.py`
- [x] `XOR_X` / `XOR_Y` sowie `SEVEN_SEG_X` / `SEVEN_SEG_Y` (Ziffern 0–9, Tabelle von Seite 10
  der Angabe) — behebt Schwachstelle 6

### Nachweis — `tests/test_kern.py`, 28 Tests, alle grün

| Prüfung | Ergebnis |
|---|---|
| Gradient-Check Backprop vs. zentrale Differenz, 5 Architekturen inkl. 4-16-7 | max. Abweichung 7·10⁻¹¹ (Grenze 10⁻⁶) |
| Gegenprobe: absichtlich eingebaute Fehler | beide erkannt (5·10⁻³ und 0,74) |
| Neues Paket vs. Ausgangsstand, XOR, 20000 Epochen | Kostenverlauf und Gewichte **bit-identisch** |
| `python neural_network.py` vs. Original | Ausgabe identisch (0.037759 / 0.958389 / 0.958241 / 0.052827) |
| `to_dict` → JSON → `from_dict` | Vorhersagen bit-identisch, keine Zufallszahlen verbraucht |
| Datensätze gegen die Angabe | 7-Segment-Eingaben = Binärdarstellung der Ziffer, alle Muster verschieden |
| Falsche Daten und Strukturen | klare `ValueError`-Meldungen |

---

## Phase 2 — Part 1, Steps 4–6: Persistenz ✅

Umgesetzt in `nn/persistence.py`. Entscheidungen: [CLAUDE.md](CLAUDE.md), E-14 bis E-17.

> **Abweichung vom Plan:** Zahlen werden nicht mit `%.6f`, sondern mit `repr` geschrieben (E-14) —
> dadurch sind alle drei Formate bit-exakt. Der Header heißt `<name>_net.h` und kopiert die Werte
> über `nn_alloc()` in das Netz (E-16). Ergebnis: Abweichung nach dem Laden **0**.

### Step 4 — JSON

```json
{
  "layer_sizes": [4, 16, 7],
  "learning_rate": 0.1,
  "layers": [
    { "activation": "sigmoid",
      "neurons": [ { "bias": 0.1234, "weights": [ ... ] } ] }
  ]
}
```

`dump_json(net, pfad)` und `load(pfad) -> NeuralNetwork`. Floats über `repr` serialisiert,
damit der Roundtrip **bit-exakt** ist.

### Step 5 — TXT

Hier liegt die zentrale Design-Entscheidung dieser Phase: **dieselbe Datei muss
menschenlesbar sein (Step 5) und gleichzeitig vom C-Code geparst werden (Part 4).**

Lösung: Kommentarzeilen beginnen mit `#`, die Nutzdaten sind rein numerisch.

```
# Neuronales Netz - 7-Segment-Decoder
# Struktur: 4 -> 16 -> 7
# Anzahl der Layer (ohne Eingangsschicht)
2
# Layer 0: <neuronen> <eingaenge> <aktivierung>
16 4 sigmoid
# Neuron 0: bias, danach 4 Gewichte
  -0.418239   0.771254  -0.203118   1.042660   0.338901
```

Der C-Parser liest zeilenweise mit `fgets`, überspringt Kommentar- und Leerzeilen und wertet
den Rest mit `sscanf` aus. Zusätzlich gibt es `load_txt()` in Python — damit lässt sich der
C-Parser direkt gegen die Python-Referenz prüfen.

### Step 6 — Header (`.h`)

Erzeugt `models/7seg_net.h`:

- Include-Guard, `#define NN_NUM_LAYERS`, `NN_LAYER_SIZES[]`
- pro Layer `static const double NN_L0_BIASES[]` und `NN_L0_WEIGHTS[][]`
- `nn_init_from_header(Network *net)` zum Befüllen der Structs

Der Header inkludiert `nn_forward.h` aus Part 4, damit Part 5 dieselben Datenstrukturen
verwendet und nichts dupliziert wird.

---

## Phase 3 — Part 2: Training und Persistenz ✅

> **Ergebnis:** XOR 4/4, 7-Segment 70/70 Bits und 10/10 Ziffern (Seed 1, 33 s). Die Referenz für alle
> späteren Vergleiche sind die Vorhersagen direkt nach dem Training in `models/*_training.json` (E-18).
> Gemeinsame Auswertung (Schwellwert, Accuracy, ASCII-Anzeige) in `auswertung.py`.

| Skript | Angabe | Inhalt |
|---|---|---|
| `train_xor.py` | Part 2, Step 1 | 2-2-1, 20000 Epochen, α = 0.1 → `models/xor.json`, `.txt`, `.h` |
| `test_xor_load.py` | Part 2, Step 2 | JSON laden, dieselben 4 Eingaben testen, **maximale Abweichung** zum Original ausgeben (Toleranz 1e-12), Accuracy mit Schwellwert 0.5 |
| `train_7seg.py` | Part 2, Step 3 | 4-16-7, 20000 Epochen, α = 0.1, 10 Samples → `models/7seg.json`, `.txt`, `7seg_net.h` |
| `test_7seg_load.py` | Part 2, Step 4 | Ziffern 0–9 prüfen, Bit- und Ziffern-Accuracy, zusätzlich ASCII-Rendering der Ziffer zur Sichtkontrolle |

Die bestehende Ergebnistabelle aus `train_xor()` wird übernommen. Laufzeit des
7-Segment-Trainings: geschätzt rund eine Minute, Fortschrittsausgabe alle 2000 Epochen.

---

## Phase 4 — Part 3: Vergleich mit TensorFlow ✅

> **Ergebnis** (E-20 bis E-22): TensorFlow 2.21 in `.venv-tf` (Python 3.12). Mit identischen
> Startgewichten stimmt TF mit der eigenen Implementierung auf 1,4·10⁻¹⁶ überein — ein unabhängiger
> Nachweis der Backpropagation. Keras-Standard bleibt bei XOR in einem lokalen Minimum (3/4).
> Tabelle in [README.md](README.md), Rohdaten in `ergebnisse/tf_vergleich.json`.

1. Python 3.12 installieren, venv `.venv-tf` anlegen, dort `tensorflow` installieren.
   Das Haupt-Python 3.14 bleibt unangetastet (siehe [CLAUDE.md](CLAUDE.md), E-03).
2. `compare_tensorflow.py`: XOR (2-2-1) und 7-Segment, jeweils Sigmoid-Aktivierung,
   MSE-Loss, SGD, gleiche Epochenzahl und Lernrate wie in der eigenen Implementierung.
3. **Offener Punkt aus der Angabe:** Part 2 nennt 16 Hidden-Neuronen, Part 3 für TensorFlow
   aber 10. Auflösung: **beide** Varianten trainieren — 16 für den fairen 1:1-Vergleich,
   10 zusätzlich, weil die Angabe es so vorgibt.
4. Vergleichstabelle: Prediction je Eingabe, finaler MSE, Accuracy, Trainingsdauer in
   Sekunden, Epochen bis Kosten < 0.01.

---

## Phase 5 — Part 4: Forward-Only in C, Initialisierung aus `.txt` ✅

> **Ergebnis** (E-19): MinGW-w64 (gcc 16.2 via winget), 70/70, Abweichung zu Python 2,2·10⁻¹⁶,
> ohne Warnung mit `-Wall -Wextra -Werror`. Zusätzlich zum Plan: `test_7seg.c` (gemeinsamer
> Testlauf für Part 4 und 5), `--roh` für den automatischen Vergleich, Exit-Codes 0/1/2.

1. MinGW-w64 installieren, `gcc` prüfen (siehe [CLAUDE.md](CLAUDE.md), E-04).
2. `c_forward/nn_forward.h` — Structs:
   `Neuron { double *weights; int num_inputs; double bias; double output; }`,
   darauf aufbauend `Layer` und `Network`.
3. `nn_forward.c` — `nn_load_txt()` (Parser nach dem Format aus Phase 2), `nn_forward()`,
   `nn_predict()`, `nn_free()`, `sigmoid()`. **Kein Training**, wie in der Angabe gefordert.
4. `main_txt.c` — lädt `models/7seg.txt`, testet `[0,0,0,0]` bis `[1,0,0,1]`,
   gibt Ist- und Soll-Werte aus.
5. `Makefile` mit `-Wall -Wextra -std=c11`.

**Nachweis:** Die C-Ausgaben müssen mit denen von `test_7seg_load.py` auf ~1e-12
übereinstimmen.

---

## Phase 6 — Part 5: Forward-Only in C, Initialisierung aus `.h` ✅

> **Ergebnis:** 70/70, Abweichung zu Python 2,2·10⁻¹⁶, `nn_forward.c` und `test_7seg.c` werden aus
> `c_forward/` mitkompiliert.

`c_header_init/main_header.c` nutzt **dieselben** Structs und dieselbe `nn_forward()` wie
Part 4 — `nn_forward.c` wird einfach mitkompiliert, kein Copy-Paste. Der Unterschied liegt
allein in der Initialisierung: `nn_init_from_header()` aus `models/7seg_net.h` statt
Datei-Parsing. Gleicher Testlauf, gleicher Nachweis.

---

## Phase 7 — ESP32-Ersatz: interaktive Visualisierung ✅

> **Ergebnis** (E-23): `visualisierung/index.html`, erzeugt von `export_daten.py` aus
> `vorlage.html`. Alle geplanten Punkte umgesetzt, dazu: Neuron-Inspektor, Gewichts-Heatmaps,
> Übersicht aller 16 Eingaben, Backpropagation mit Probeschritten, Trainings- und TF-Kurven,
> Live-Training im Browser, XOR-Entscheidungsflächen, Experimente und ein Nachweis-Kapitel:
> Browser ↔ Python 2,2·10⁻¹⁶, Browser ↔ C bit-identisch.

Das Herzstück der Planänderung. Eine interaktive HTML-Seite, die zeigt, **wie** das Netz
zu seiner Ausgabe kommt — nicht nur, dass die Ausgabe stimmt.

- Ein Export-Skript schreibt die Gewichte aus `models/7seg.json` in die Seite; der
  Forward-Pass läuft in JavaScript. Es rechnet also **tatsächlich das trainierte Netz**,
  keine hartkodierte Wahrheitstabelle — sonst wäre die Visualisierung wertlos.
- **Eingabe:** Ziffer 0–9 wählen oder die 4 Eingangsbits einzeln umschalten.
- **Netzgraph** 4 → 16 → 7: Neuronen nach Aktivierungswert eingefärbt, Kanten nach Gewicht
  (Farbe = Vorzeichen, Dicke = Betrag).
- **Schritt-für-Schritt-Modus** — der Kern laut Auftrag:
  Eingabe → `Z1 = W1·x + b1` → `A1 = sigmoid(Z1)` → `Z2 = W2·A1 + b2` → `A2 = sigmoid(Z2)`
  → Schwellwert → Segmente. Zu jedem Schritt die konkreten Zahlen des gerade gewählten
  Neurons.
- **Ausgabe** als 7-Segment-Display, Segmenthelligkeit proportional zum Aktivierungswert —
  so werden Unsicherheiten des Netzes unmittelbar sichtbar. Daneben die Rohwerte.

---

## Phase 8 — Erweiterungen: von gut zu exzellent ✅

Alle acht Punkte umgesetzt, Messwerte in [CLAUDE.md](CLAUDE.md) E-24 bis E-27:

| # | Umsetzung | Ergebnis |
|---|---|---|
| 1 | `tests/test_persistenz.py`, `tests/test_c.py` | 55 Tests grün (vorher 28) |
| 2 | `experimente/aktivierungen.py` | 7-Segment: tanh/ReLU 4–6× schneller; XOR 2-2-1 scheitert damit meist |
| 3 | `train(batch_size=, momentum=)` | Momentum 0,9: XOR fast 10× schneller; Standard bleibt bit-identisch |
| 4 | `experimente/rauschen.py` | Training mit Rauschen: 95,6 % statt 78,5 % Ziffern bei σ = 0,2 |
| 5 | `experimente/seed_studie.py` (+ `parallel.py`) | ab 2-3-1 keine lokalen Minima mehr |
| 6 | `experimente/vektorisiert.py` | 3,8× (4-16-7) bis 9,7× (4-64-7), bei XOR langsamer |
| 7 | `experimente/plots.py` | `ergebnisse/verlauf_eigen.png`, `verlauf_tensorflow.png` |
| 8 | `README.md` | Reproduktionsanleitung und Ergebnisübersicht |

Ursprüngliche Planung (Reihenfolge eingehalten):

| # | Erweiterung | Bezug zur Angabe |
|---|---|---|
| 1 | **Testsuite** (`pytest`) ausbauen: Grundstock steht seit Phase 1 (`tests/test_kern.py`: Gradient-Check, Ableitungen, Gleichwertigkeit, `to_dict`). Dazu kommen JSON/TXT/.h-Roundtrip, 7-Segment-Accuracy und der Abgleich Python ↔ C | explizite Bonus-Aufgabe |
| 2 | **ReLU und tanh** durchgängig nutzbar, Konvergenzvergleich | explizite Bonus-Aufgabe |
| 3 | **Mini-Batch und Momentum** als Optionen im Trainer | explizite Bonus-Aufgabe |
| 4 | **Rausch-Robustheit**: 7-Segment mit gekippten Eingangsbits testen | optionale Erweiterung, Seite 11 |
| 5 | **Seed-Studie XOR** als reproduzierbares Skript mit Grafik. Eine Vorab-Messung in Phase 1 (E-11) hat die ursprüngliche These *widerlegt*: `klein` ist bei 2-2-1 zuverlässiger als Xavier. Ausbau: Erfolgsquote über 2, 3 und 4 Hidden-Neuronen × beide Initialisierungen — zeigt, ab welcher Netzgröße die lokalen Minima verschwinden | eigene Ergänzung |
| 6 | **Vektorisierte Variante** als Benchmark gegen die Neuron-für-Neuron-Implementierung | eigene Ergänzung |
| 7 | **Loss-Kurven** (matplotlib): XOR vs. 7-Segment, eigene Implementierung vs. TensorFlow | optionale Erweiterung, Seite 11 |
| 8 | `README.md` mit vollständiger Reproduktionsanleitung | eigene Ergänzung |

---

## Phase 9 — Ziffern zeichnen und erkennen ✅

> Wunsch des Users (19.10.2026): eine "coolere" Erweiterung. Entscheidungen: E-29 bis E-33.

| # | Umsetzung | Ergebnis |
|---|---|---|
| 1 | Softmax (Schicht-Aktivierung) in `nn/activations.py` und `nn/layer.py` | Neuron/Layer-Struktur bleibt; alte Netze bit-identisch |
| 2 | `loss="cross_entropy"` in `nn/network.py` | delta = A − Y direkt; `"loss"` nur in der JSON, wenn nicht MSE |
| 3 | `tests/test_softmax.py` | 13 Tests, Gradient-Check < 10⁻⁶; insgesamt 70 Tests grün |
| 4 | `train_digits.py` | 196-64-10 auf MNIST 14×14, 97,7 % auf 10 000 Testbildern, ~5 min (zuerst UCI 8×8 mit 96,7 %, ersetzt: E-36) |
| 5 | Kapitel "Zeichnen" in `visualisierung/vorlage.html` | live: Zeichnung → MNIST-Aufbereitung → 14×14 → Hidden-Bilder → Wahrscheinlichkeiten |

---

## Phase 10 — Neugestaltung der Visualisierung ✅

> Wunsch des Users (19.10.2026): Seite "viel übersichtlicher". Entscheidung: E-35.

- Fünf Bereiche als Tabs (Decoder · Zeichnen · So rechnet es · Training · Ergebnisse), immer nur
  ein Kapitel sichtbar, Kapitel per Adresse verlinkbar
- Eingabe-Leiste unter "So rechnet es", aufklappbare Zusatzregler, Kennzahlen bei "Ergebnisse"
- ruhigere Optik: kompakte Kopfzeile, keine Eyebrow-Zeilen, mehr Abstand; Rechenlogik unverändert

---

## Phase 11 — Abgabe-Dokumentation mit Quarto ✅

> Wunsch des Users (19.10.2026). Entscheidung: E-37.

- `dokumentation/dokumentation.qmd` → HTML (eigenständig, Formeln als MathML) und PDF (Typst, 26 Seiten)
- Inhalt: Aufgabenstellung, Theorie mit allen Formeln der Angabe, Python-Code, Exportformate, Training,
  TensorFlow, C, Visualisierung (12 Screenshots), Bonus und Experimente, MNIST, Tests, Fazit, Anhang
- reproduzierbar: `screenshots.py` (Edge headless), `abbildungen.py` (matplotlib), `quarto render`

---

## Verifikation

```
py -3.14 train_xor.py          # -> models/xor.json/.txt/.h,  4/4 korrekt
py -3.14 test_xor_load.py      # -> maximale Abweichung < 1e-12
py -3.14 train_7seg.py         # -> models/7seg.json/.txt/.h, 10/10 Ziffern
py -3.14 test_7seg_load.py     # -> 70/70 Bits korrekt
py -3.14 train_digits.py       # -> models/digits.json, 97,7 % Test (Phase 9, braucht MNIST)
py -3.14 -m pytest -v          # -> 70 Tests gruen

.venv-tf\Scripts\python compare_tensorflow.py       # -> Vergleichstabelle

cd c_forward     && mingw32-make && nn_txt ..\models\7seg.txt   # 10/10, Werte == Python
cd c_header_init && mingw32-make && nn_header                   # 10/10, Werte == Python

py -3.14 visualisierung/export_daten.py                         # -> visualisierung/index.html
```

**Das härteste Kriterium:** Dieselben Eingaben müssen in Python, in C (aus `.txt`), in C
(aus `.h`) und in der HTML-Visualisierung identische Ausgaben liefern — Abweichung ≤ 1e-9.
Damit ist bewiesen, dass alle drei Exportformate verlustfrei sind. Genau darum geht es in
den Parts 2, 4 und 5.

---

## Reihenfolge und Abhängigkeiten

Phase 0 → 1 → 2 → 3 sind **strikt sequentiell**: alles Weitere baut auf den exportierten
Dateien des 7-Segment-Netzes auf.

Danach sind Phase 4 (TensorFlow), Phase 5 + 6 (C) und Phase 7 (Visualisierung)
**unabhängig voneinander** und können in beliebiger Reihenfolge abgearbeitet werden.

Die beiden Installationen (Python 3.12, MinGW-w64) werden erst angestoßen, wenn Phase 3
steht — sie blockieren den Hauptteil nicht.
