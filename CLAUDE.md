# CLAUDE.md — Entscheidungs- und Arbeitsprotokoll

Dieses Dokument wird **laufend** mitgeführt. Jede wichtige Entscheidung, jeder relevante
Gedankenschritt und jede Abweichung vom ursprünglichen Plan wird hier festgehalten —
nicht nachträglich rekonstruiert, sondern direkt beim Treffen der Entscheidung.

**Format jedes Eintrags:** Entscheidung → Begründung → verworfene Alternative(n).

---

## Verbindliche Arbeitsregeln

Diese Regeln gelten für **jede** Code-Änderung im Projekt — Python, C und JavaScript —
ohne dass sie jedes Mal neu verlangt werden müssen.

### R-01: Code wird immer so kommentiert, dass ein Mensch ihn im Nachhinein versteht

Maßstab: Jemand, der die Angabe kennt, aber den Code nie gesehen hat — oder der Autor selbst
Monate später — muss ohne Rückfrage verstehen können, **was** der Code macht und **warum**
er es so macht.

Konkret heißt das:

1. **Dateikopf:** Jede Datei beginnt mit einem Docstring bzw. Kommentarblock: Zweck der Datei,
   welchen Part/Step der Angabe sie umsetzt, wie man sie ausführt.
2. **Jede Klasse und jede Funktion** bekommt einen Docstring: was sie tut, Parameter mit
   Bedeutung und Form (Skalar, Vektor, Länge), Rückgabewert, Seiteneffekte
   (z. B. "verändert die Gewichte").
3. **Mathematik wird an die Angabe angebunden:** Wo eine Formel umgesetzt wird, steht sie im
   Kommentar in der Notation der Angabe dabei (z. B. `dJ/dA[2] = A[2] - Y`). So lässt sich
   jede Codezeile gegen das PDF prüfen.
4. **Warum vor Was:** Nicht-offensichtliche Stellen erklären den Grund, nicht nur die Tätigkeit.
   Ein Kommentar wie `# Gewichte aktualisieren` über `update()` ist wertlos; wertvoll ist
   `# erst NACH der Fehlerberechnung, sonst rechnet die Kettenregel mit neuen Gewichten`.
5. **Formen und Einheiten:** Bei Arrays steht dabei, wie lang sie sind und wofür jeder Eintrag
   steht (z. B. "Länge = Anzahl Eingänge dieses Neurons").
6. **Fallstricke markieren:** Stellen, an denen eine naheliegende Änderung den Code kaputt
   machen würde, bekommen ein deutliches `WICHTIG:`.
7. **C-Code zusätzlich:** Speicherbesitz (wer `malloc`t, wer `free`t) und Rückgabewerte
   im Fehlerfall werden immer dokumentiert.
8. **Kommentare wandern mit:** Wird Code geändert, werden die zugehörigen Kommentare im selben
   Schritt angepasst. Ein veralteter Kommentar ist schlimmer als keiner.
9. **Nur Kommentare heißt nur Kommentare:** Wird eine Datei rein zur Dokumentation
   überarbeitet, bleibt die Logik bit-identisch — nachgewiesen durch Vergleich der Ausgabe
   vorher/nachher.

Sprache der Kommentare: Deutsch (siehe E-06).

### R-02: Jeder Prompt wird in `Prompts.md` protokolliert

Jeder Prompt des Users bekommt in [Prompts.md](Prompts.md) einen Eintrag `P-xx` mit Datum,
einer **gekürzten** Fassung des Prompts und dem Ergebnis. Rückfragen und die Antworten darauf
gehören mit hinein. Der Eintrag wird zu Beginn der Arbeit angelegt und am Ende mit dem
Ergebnis ergänzt.

Abgrenzung: `Prompts.md` sagt, **was** verlangt und geliefert wurde. `CLAUDE.md` sagt,
**warum** etwas so entschieden wurde.

---

## 2026-09-29 — Projektstart, Bestandsaufnahme und Rahmenbedingungen

### E-01: Planänderung — Part 6 (ESP32) entfällt

**Entscheidung:** Part 6 der Angabe (Portierung auf ESP32, physisches 7-Segment-Display,
Ansteuerung über Serial Monitor) wird **nicht** umgesetzt. Stattdessen entsteht eine
**interaktive Visualisierung** des 7-Segment-Decoders.

**Begründung:** Vorgabe des Users. Kein physisches ESP32-Board im Spiel. Die Visualisierung
erfüllt denselben didaktischen Zweck — sichtbar machen, dass das Netz tatsächlich dekodiert —
und ist zusätzlich vorzeigbar, ohne Hardware aufbauen zu müssen.

**Verworfen:** Simulation des ESP32 (z. B. Wokwi). Hätte Aufwand in die Emulation gesteckt
statt in die Nachvollziehbarkeit des Netzes selbst.

### E-02: Form der Visualisierung — interaktive HTML-Seite

**Entscheidung:** Eine interaktive HTML-Seite. Die Gewichte des trainierten Netzes werden aus
`models/7seg.json` in die Seite exportiert, der Forward-Pass läuft in JavaScript.

**Begründung:** Es muss **wirklich das trainierte Netz** rechnen, keine hartkodierte
Wahrheitstabelle — sonst ist die Visualisierung wertlos. Eine HTML-Seite lässt sich ohne
Installation weitergeben und vorführen.

**Verworfen:** Tkinter/Matplotlib-GUI (nur lokal lauffähig, schlecht weiterzugeben);
statische PNGs (nicht interaktiv, kein Schritt-für-Schritt-Nachvollzug).

### E-03: TensorFlow (Part 3) — separates Python 3.12 + venv

**Befund:** Installiert ist Python 3.14.2. TensorFlow unterstützt maximal Python 3.12/3.13,
`pip install tensorflow` schlägt unter 3.14 fehl.

**Entscheidung:** Python 3.12 zusätzlich installieren, venv `.venv-tf` anlegen, TensorFlow nur
dort installieren. Das Haupt-Python 3.14 bleibt unangetastet.

**Begründung:** Part 3 verlangt explizit TensorFlow. Ein Downgrade des Haupt-Python würde den
Rest des Projekts gefährden; ein venv isoliert sauber.

**Verworfen:** PyTorch oder `sklearn.MLPClassifier` als Ersatz — weicht von der Angabe ab und
macht den geforderten Vergleich angreifbar.

### E-04: C-Compiler (Parts 4/5) — MinGW-w64 installieren

**Befund:** Kein `gcc`, `clang`, `cl` oder MinGW auf dem System gefunden.

**Entscheidung:** MinGW-w64 installieren, damit der C-Code tatsächlich kompiliert **und gegen
die Python-Ergebnisse verifiziert** werden kann.

**Begründung:** Nicht kompilierter C-Code in einer Abgabe ist ein unnötiges Risiko — der
zentrale Nachweis von Parts 4/5 ist ja gerade, dass Export und Re-Import verlustfrei sind.
Das lässt sich nur durch Ausführen belegen.

**Verworfen:** C-Code schreiben ohne zu kompilieren.

### E-05: Code-Struktur — modulares Paket statt Einzeldatei

**Entscheidung:** Der Kern wandert aus `neural_network.py` in ein Paket `nn/`
(`activations`, `neuron`, `layer`, `network`, `persistence`), Trainings- und Testskripte
liegen als eigene Dateien im Projekt-Root.

**Begründung:** Die Angabe umfasst 5 Parts mit je eigenen Ein-/Ausstiegspunkten. In einer
einzigen Datei wird das unübersichtlich und schlecht benotbar. Skripte bleiben im Root, damit
`import nn` ohne `sys.path`-Manipulation funktioniert.

**Verworfen:** Alles in `neural_network.py` lassen; eine Datei pro Part mit dupliziertem Kern.

### E-06: Dokumentationssprache — Deutsch

**Entscheidung:** Alle Markdown-Dokumente, Docstrings, Kommentare und Konsolenausgaben auf
Deutsch. Bezeichner im Code (Klassen, Methoden) bleiben englisch.

**Begründung:** Der vorhandene Code ist bereits deutsch kommentiert; die Abgabe ist deutsch.
Englische Bezeichner, weil die Angabe die Namen `Neuron`, `Layer`, `forward`, `predict`,
`dump_json` usw. wörtlich vorgibt.

### E-07: numpy bleibt erlaubt

**Entscheidung:** numpy wird weiter verwendet — aber ausschließlich für Vektor-/Skalarprodukte
und Zufallszahlen, **nicht** für fertige NN-Bausteine oder automatische Differentiation.

**Begründung:** "From scratch" meint Forward-Pass, Backpropagation und Gradientenabstieg selbst
implementieren. Das ist erfüllt: jede Ableitung ist im Code explizit hingeschrieben. `np.dot`
ersetzt lediglich eine handgeschriebene Schleife.

---

## 2026-09-29 — Phase 1: Kern in das Paket `nn/` überführt

### E-08: Ausgangsstand archivieren statt überschreiben

**Entscheidung:** Die kommentierte Originaldatei liegt unverändert in
`archiv/neural_network_ausgangsstand.py`. `neural_network.py` bleibt als schlanker
Einstiegspunkt bestehen: Sie importiert aus `nn/` und trainiert beim Start XOR.

**Begründung:** `Ausgangsstand.md` verweist mit Zeilennummern auf den alten Code. Würde die
Datei umgebaut, zeigten alle Verweise ins Leere. Außerdem ist die Archivdatei der Maßstab für
den Gleichwertigkeitstest (E-13). Der Einstiegspunkt bleibt, damit `python neural_network.py`
und `from neural_network import NeuralNetwork` weiter funktionieren.

**Verworfen:** Den alten Stand nur über ein Backup im Temp-Ordner festhalten — dann wäre der
Nachweis nicht reproduzierbar.

### E-09: Aktivierung über Namen, Ableitung als `derivative(z, a)`

**Entscheidung:** `nn/activations.py` enthält eine Registry `Name → (Funktion, Ableitung)` für
`sigmoid`, `tanh` und `relu`. Neuronen speichern den Namen. Alle Ableitungen haben die
Signatur `derivative(z, a)`.

**Begründung:** Ein Name lässt sich nach JSON schreiben, ein Funktionsobjekt nicht — das war
der Blocker für Part 1 Step 4. Die Signatur mit `z` *und* `a` erlaubt, dass Sigmoid und tanh
die bereits gespeicherte Ausgabe `a` nutzen (`a·(1−a)`, genau die Formel der Angabe) und
ReLU trotzdem `z` bekommt. `a` allein reicht bei ReLU nicht, weil `a = 0` nicht verrät, ob
`z` negativ oder genau 0 war.

**Nachweis:** Das Ergebnis ist bit-identisch mit dem alten `sigmoid(z)·(1−sigmoid(z))`, weil
`a` exakt mit derselben Rechnung entstanden ist (siehe Test in E-13).

### E-10: Korrektur — die Formel im Plan war nicht Xavier

**Befund:** Plan und `Implementation.md` nannten als Xavier-Initialisierung
`randn · sqrt(1/n_in)`. Das ist streng genommen die LeCun-Initialisierung. Xavier/Glorot ist
`randn · sqrt(2/(n_in + n_out))` und braucht auch die Größe der eigenen Schicht.

**Entscheidung:** Implementiert ist die echte Glorot-Formel. Deshalb erzeugt der **Layer** die
Startgewichte und nicht das einzelne Neuron: Nur der Layer kennt `n_out`.

**Nebeneffekt, bewusst so gebaut:** Der Layer zieht die Zufallszahlen Neuron für Neuron in
derselben Reihenfolge wie der Ausgangsstand. Mit `init="klein"` und gleichem Seed entstehen
dadurch exakt dieselben Startgewichte wie früher.

### E-11: Standard-Initialisierung `klein` statt `xavier` — nach Messung

**Ausgangslage:** Der Plan wollte Xavier als Standard einführen, begründet mit
Schwachstelle 5 aus `Ausgangsstand.md`: Kleine Startgewichte ließen XOR in lokalen Minima
hängen. Das war eine Annahme und wurde vor der Entscheidung gemessen.

**Messung 1 — XOR 2-2-1, 20000 Epochen, α = 0.1, je 100 Seeds:**

| | `klein` (0.1) | `xavier` |
|---|---|---|
| 4/4 richtig nach 20000 Epochen | 86 | 75 |
| sauber (jede Ausgabe < 0.1 vom Ziel) | 58 | 74 |
| Kosten < 0.01 erreicht, Median-Epoche | 15919 | 5790 |

**Messung 2 — die Fehlschläge aus Messung 1, weitertrainiert bis 80000 Epochen:**

| | `klein` | `xavier` |
|---|---|---|
| waren nur langsam, danach 4/4 | 7 | 1 |
| hängen **dauerhaft** fest | **7** | **24** |

Die dauerhaften Fehlschläge sind echte lokale Minima mit konstanten Kosten. Typische
Ausgaben sind `[0.01, 0.5, 0.99, 0.5]` (Kosten ≈ 0.063) oder `[0.02, 0.67, 0.67, 0.67]`
(Kosten ≈ 0.084): Eines der beiden Hidden-Neuronen hat sich festgelegt, bevor das andere
seine Rolle gefunden hat.

**Messung 3 — 7-Segment 4-16-7, 20000 Epochen, je 20 Seeds:** beide Verfahren 20/20 mit
70/70 Bits richtig, Median bis Kosten < 0.01: 2678 (`klein`) gegenüber 2453 (`xavier`),
praktisch gleich.

**Entscheidung:** Standard bleibt `klein`, `xavier` ist als Option verfügbar.

**Begründung:** Bei XOR ist `klein` deutlich zuverlässiger (7 statt 24 von 100 dauerhaft
hängend), beim 7-Segment-Netz gibt es keinen Unterschied. Xavier ist schneller, *wenn* es
klappt — für eine Abgabe zählt aber, dass das Netz lernt. Als Bonus ist das Verhalten ohne
`init`-Angabe identisch mit dem Ausgangsstand.

**Erklärung:** Mit kleinen Gewichten starten beide Hidden-Neuronen fast gleich und nahe dem
symmetrischen Sattelpunkt. Sie lösen sich langsam, aber meist in Richtung der richtigen
Lösung. Mit großen Gewichten ist die Rollenverteilung schon beim Start "ausgewürfelt" — oft
ungünstig, und mit nur 2 Hidden-Neuronen gibt es keine Reserve, die das ausgleicht.

**Folgen:** Schwachstelle 5 in `Ausgangsstand.md` ist durchgestrichen und korrigiert. Der
Kommentar in der Archivdatei ist angepasst, mit gleicher Zeilenzahl, damit die Verweise
stimmen. Die geplante Seed-Studie in Phase 8 wird umformuliert (siehe `Implementation.md`).

**Verworfen:** Xavier trotzdem als Standard, weil es "Best Practice" ist — die Messung zeigt,
dass das für diese Netzgröße nicht gilt.

### E-12: `to_dict()`/`from_dict()` auf allen drei Ebenen schon in Phase 1

**Entscheidung:** Neben `Neuron` (so im Plan) haben auch `Layer` und `NeuralNetwork`
`to_dict()`/`from_dict()`. `nn/persistence.py` (Phase 2) muss dann nur noch Dateien schreiben
und lesen.

**Begründung:** Jede Klasse weiß selbst am besten, was sie ausmacht. `from_dict` prüft beim
Laden, dass alle Größen zusammenpassen, und verbraucht keine Zufallszahlen: Ein geladenes Netz
verändert den Zufallszustand des Programms nicht.

**Kleinere Festlegungen:**
- `train(..., shuffle=False)`: mischen ist möglich, aber aus. Die Angabe beschreibt eine feste
  Reihenfolge, und nur so bleibt der Lauf bit-identisch mit dem Ausgangsstand.
- `train()` prüft die Trainingsdaten vorab, der Konstruktor die Struktur. Ein Tippfehler in
  den Daten führt zu einer klaren Fehlermeldung statt zu einem kryptischen numpy-Fehler.
- `backward()` der ersten Schicht berechnet keinen Fehler mehr für die Eingabe (Schwachstelle 4).

### E-13: Wie Phase 1 verifiziert wird

**Entscheidung:** drei unabhängige Nachweise in `tests/test_kern.py` (28 Tests, `pytest`):

1. **Gradient-Check:** Die Gradienten der Backpropagation werden mit einem numerischen
   Gradienten (zentrale Differenz) verglichen, auf 5 Architekturen inklusive 4-16-7 und einem
   Netz mit zwei Hidden-Schichten. Die analytischen Gradienten werden ohne Umbau des Codes
   gewonnen: `backward()` auf einer Kopie mit α = 1 ausführen, dann gilt
   `dJ/dW = W_alt − W_neu`. So wird genau der Code geprüft, der auch trainiert.
   Maximale Abweichung: 7·10⁻¹¹, Grenze 10⁻⁶.
2. **Gleichwertigkeit:** Das neue Paket und die Archivdatei trainieren XOR mit Seed 1 über
   20000 Epochen. Kostenverlauf und alle Gewichte sind **bit-identisch** (`==`, nicht
   "ungefähr").
3. **Gegenprobe:** Der Gradient-Check wurde gegen zwei absichtlich eingebaute Fehler
   getestet und erkennt beide: "Update vor Fehlerberechnung" (Abweichung 5·10⁻³) und
   "Ableitung vergessen" (0,74). Ein Test, der Fehler nicht erkennen kann, wäre wertlos.

---

## 2026-10-12 — Phase 2: Persistenz (`nn/persistence.py`)

### E-14: Alle Formate schreiben Zahlen mit `repr` — nicht mit `%.6f`

**Entscheidung:** JSON, TXT und Header schreiben jede Gleitkommazahl als `repr(float)`, die
kürzeste Dezimaldarstellung, die beim Einlesen wieder exakt denselben double ergibt.

**Begründung:** Python `float()`, C `strtod()` und der C-Compiler bei Literalen runden
Dezimalzahlen alle korrekt. Mit `repr` haben Python, C aus `.txt` und C aus `.h` deshalb
bit-identische Gewichte. Gemessen: Abweichung nach dem Laden **exakt 0** (XOR und 7-Segment,
JSON und TXT). Die TXT bleibt trotzdem lesbar: Spalten ausgerichtet, jede Zahl erklärt.

**Verworfen:** `%.6f` (im Plan skizziert) — lesbarer, aber bis zu 5·10⁻⁷ Fehler pro Gewicht;
der Nachweis "Export ist verlustfrei" wäre dann nur noch "ungefähr" möglich.

### E-15: TXT-Format — Kommentarzeilen und rein numerische Datenzeilen, ASCII

**Entscheidung:** Zeilen mit `#` sind Kommentar, alle anderen enthalten nur Zahlen bzw. die
Kopfzeile `<Neuronen> <Eingänge> <Aktivierung>`. Die Neuronennummer steht in einer eigenen
Kommentarzeile, nicht hinter den Zahlen. Die Datei ist reines ASCII ("Eingaenge").

**Begründung:** Eine Regel macht die Datei für Menschen (Step 5) und für den C-Parser (Part 4)
tauglich. Ohne Kommentare mitten in der Zeile reicht im C-Code `strtod` in einer Schleife.
ASCII erspart dem C-Code jede Frage nach der Zeichenkodierung.

### E-16: Header kopiert die Konstanten über `nn_alloc()` statt Zeiger darauf zu setzen

**Entscheidung:** Der Header enthält `static const`-Arrays und `nn_init_from_header()`, das
über `nn_alloc()` aus `nn_forward.c` Speicher anlegt und die Werte hineinkopiert.

**Begründung:** (1) Ein Netz aus `.txt` und eines aus `.h` sind danach gleich gebaut und
werden gleich mit `nn_free()` freigegeben — kein Sonderfall im Speicherbesitz. (2) Zeiger auf
`const`-Arrays in `double *weights` zu legen, hieße `const` wegcasten. Alles im Header ist
`static`, damit er in mehreren `.c`-Dateien eingebunden werden könnte.

**Verworfen:** Neuron-Structs direkt im Header als Initialisierer (ohne Heap) — schneller beim
Start, aber dann bräuchte Part 5 eine zweite Netz-Struktur oder eine Sonderregel in `nn_free`.

### E-17: `dump_json()` usw. zusätzlich als Methoden am Netz

**Entscheidung:** Die Arbeit steckt in `persistence.py`; `NeuralNetwork.dump_json()`,
`.load()`, `.dump_txt()`, `.load_txt()`, `.dump_initfile()` leiten nur weiter (Import in der
Methode, um einen Zirkelimport zu vermeiden).

**Begründung:** Die Angabe nennt sie als Methoden der Klasse; gleichzeitig bleibt der
Netzcode frei von Dateiformat-Details.

## 2026-10-12 — Phase 3: Training und Persistenz (Part 2)

### E-18: Referenz = Vorhersagen direkt nach dem Training, gespeichert in `*_training.json`

**Entscheidung:** `train_*.py` speichert neben dem Netz Seed, Dauer, Kostenverlauf und die
Vorhersagen unmittelbar nach dem Training. `test_*_load.py`, die C-Tests, die Visualisierung
und `export_daten.py` vergleichen gegen genau diese Werte.

**Begründung:** "Verify that the predictions are similar to the outputs from the trained model"
verlangt die Ausgaben des *trainierten* Modells — nicht eines neu trainierten. Erneutes
Trainieren in den Testskripten wäre langsam und würde nur Determinismus prüfen.

**Ergebnis:** XOR 4/4 (0,0378 / 0,9584 / 0,9582 / 0,0528, identisch mit dem Ausgangsstand),
7-Segment 70/70 Bits und 10/10 Ziffern mit Seed 1 (Kosten 4,4·10⁻⁴, kleinste Sicherheit 95,6 %).
Nach dem Laden aus JSON und TXT: Abweichung 0.

## 2026-10-12 — Phase 5/6: C (Part 4 und 5)

### E-19: C-Aufbau — Ausgabepuffer pro Layer, Zwei-Durchgang-Parser, ein gemeinsamer Test

**Entscheidungen:**
- `Layer` hat zusätzlich `double *outputs` (= A[l] als Vektor), angelegt in `nn_alloc()`.
  Dadurch kann `nn_forward()` nicht scheitern (kein `malloc` im Forward-Pass, kein NULL).
- Der Parser liest die Datei zweimal: erst nur die Kopfzeilen (Struktur prüfen, Größen
  sammeln), dann die Zahlen. `nn_alloc()` braucht alle Größen vorher.
- Der Testlauf der Ziffern 0–9 steht einmal in `c_forward/test_7seg.c`; Part 5 kompiliert
  `nn_forward.c` und `test_7seg.c` aus `c_forward/` mit — kein Copy-Paste (Part 5, Step 1).
- Exit-Codes: 0 = 70/70, 1 = falsche Segmente, 2 = Lade-/Formatfehler. `--roh` gibt nur die
  Zahlen mit `%.17g` aus, für den automatischen Vergleich in `tests/test_c.py`.

**Ergebnis:** beide Programme kompilieren mit `-Wall -Wextra -Werror` ohne Warnung, 70/70,
maximale Abweichung zu Python **2,2·10⁻¹⁶** (eine Binärstelle; numpy summiert in anderer
Reihenfolge). Kaputte Dateien (fehlend, abgeschnitten, falsche Aktivierung, falsche Zahl an
Werten) werden mit Exit-Code 2 und Zeilennummer abgelehnt.

**Verworfen:** Zwischenpuffer in `nn_forward()` per `malloc` (Fehlerfall bei jeder Vorhersage),
`realloc`-basierter Ein-Durchgang-Parser (komplizierter, kein Vorteil bei diesen Dateigrößen).

## 2026-10-12 — Phase 4: TensorFlow (Part 3)

### E-20: Offener Punkt "16 oder 10 Hidden-Neuronen" — beide trainiert

**Entscheidung (wie geplant):** 4-16-7 für den fairen 1:1-Vergleich mit Part 2, zusätzlich
4-10-7, weil Part 3 "e.g. 10" nennt. Die eigene Implementierung wird für 4-10-7 ebenfalls
trainiert, damit "same epochs and architecture" gilt.

### E-21: Zwei TensorFlow-Varianten — "identisch" und "Standard"

**Entscheidung:** *TF identisch* übernimmt die Startgewichte der eigenen Implementierung,
optimiert dieselbe Kostenfunktion ½·Σ(A−Y)² in float64. *TF Standard* nutzt die Keras-
Voreinstellungen (Glorot uniform, `MeanSquaredError`, float32).

**Begründung:** Ein Vergleich mit anderen Startgewichten sagt wenig über die Korrektheit —
bei XOR entscheidet der Startpunkt über Erfolg oder Scheitern (E-11). Mit identischem Start
muss TF Epoche für Epoche dasselbe liefern, wenn die eigene Backpropagation stimmt. TF bildet
die Gradienten per automatischer Differentiation — ein völlig unabhängiger Nachweis.

**Ergebnis:** eigen ↔ TF identisch maximal **1,4·10⁻¹⁶** in allen drei Netzen, gleiche
Konvergenz-Epoche (2-2-1: 12 494, 4-16-7: 2 626, 4-10-7: 2 991). TF Standard: XOR **3/4** —
lokales Minimum; 7-Segment 70/70, aber J < 0,01 erst nach ~9 200 Epochen, weil die Keras-MSE
durch die Anzahl der Ausgänge teilt (effektiv α/3,5). Dauer: eigen bei XOR schneller (2,6 s
gegen 11,5 s), TF bei 7-Segment (18 s gegen 33 s).

### E-22: Eigene Trainingsschleife mit `tf.function` statt `model.fit()`

**Entscheidung:** Training per `GradientTape` + `keras.optimizers.SGD` in einem kompilierten
Graphen; Kompilierzeit wird nicht mitgemessen; oneDNN aus (`TF_ENABLE_ONEDNN_OPTS=0`).

**Begründung:** `fit()` hat pro Epoche einige ms Verwaltungsaufwand; bei 20 000 Epochen à 4
Samples würde das die Zeitmessung dominieren. Gradienten und Update kommen trotzdem
vollständig von TensorFlow. oneDNN ändert die Rundung leicht und würde den Vergleich mit
"TF identisch" verrauschen.

## 2026-10-12 — Phase 7: Visualisierung

### E-23: Eine eigenständige HTML-Datei, Daten beim Export hineingeschrieben

**Entscheidung:** `visualisierung/vorlage.html` + `export_daten.py` → `visualisierung/index.html`.
Die Seite enthält Gewichte, Trainingsprotokolle, Python-Referenzwerte, C-Ausgaben,
TF-Ergebnisse und Experiment-Ergebnisse als JSON. Kein Framework, keine externen Skripte
(nur Google Fonts, mit Fallback).

**Begründung:** Per Doppelklick geöffnete Seiten (`file://`) dürfen keine Nachbardateien per
`fetch` laden. Eine Datei lässt sich weitergeben und offline vorführen.

**Kernanforderung aus E-02 erfüllt:** Der Forward-Pass läuft in JavaScript mit den exportierten
Gewichten. Die Seite rechnet beim Laden alle Ausgaben neu und vergleicht sie im Kapitel
"Nachweis" mit Python, C und TF: Browser ↔ Python 2,2·10⁻¹⁶, Browser ↔ C **bit-identisch**
(gleiche Summationsreihenfolge).

**Umfang (über den Plan hinaus):** Werkbank (Bits, Schieberegler, Rauschen; Netzgraph mit
Ansichten Gewicht / Signal w·a / Gradient), Rechenweg in 6 Schritten mit Matrix-Tabellen und
ausgeschriebener Summe, Neuron-Inspektor (Sigmoid-Kurve mit Tangente, Gewichte, Antwort auf
alle 16 Eingaben), Gewichts-Heatmaps, Übersicht aller 16 Eingaben inkl. untrainierter 10–15,
Backpropagation mit δ, Gradienten und Probeschritten auf einer Kopie, Trainingskurven und
TF-Vergleich, Live-Training im Browser, XOR-Entscheidungsflächen mit den Geraden der
Hidden-Neuronen, Experimente, Nachweis.

**Gestaltung:** LED-Anzeige als einziges kräftiges Element; Neuronen "leuchten" in derselben
LED-Farbe (Aktivierung 0..1). Vorzeichenbehaftete Größen (w, z, Gradient) in Teal/Magenta —
farbenblind-tauglich und klar getrennt von der Aktivierungsfarbe. Helles und dunkles Schema.

**Geprüft:** Headless-Edge: keine JS-Fehler beim Laden und bei simulierter Bedienung aller
Kapitel; Sichtprüfung per Screenshot.

**Verworfen:** Daten per `fetch` aus `models/` (scheitert bei `file://`); D3/Chart.js (wären
für diese Diagramme mehr Code als SVG von Hand und bräuchten Internet).

## 2026-10-12 — Phase 8: Erweiterungen

### E-24: Mini-Batch und Momentum als eigener Codepfad

**Entscheidung:** `train(..., batch_size=1, momentum=0.0)`. Nur wenn eine Option gesetzt ist,
läuft das Training über `Neuron.accumulate()` / `apply_gradient()`; sonst über den
unveränderten SGD-Pfad (`_sgd_epoch`).

**Begründung:** Ein gemeinsamer Pfad (`v = β·v − α·g; W += v` mit β = 0) rechnet
mathematisch dasselbe, aber in anderer Reihenfolge — die Bit-Gleichheit mit dem
Ausgangsstand (Test aus E-13) ginge verloren.

**Nachweis:** Full-Batch-Schritt = Mittelwert der Einzelgradienten (Abweichung < 10⁻¹⁴),
Batch 1 ohne Momentum ≈ SGD (< 10⁻¹²). **Messung XOR, Seed 1:** J < 0,01 nach 12 494
Epochen (SGD), 1 310 (Momentum 0,9), 6 191 (0,5); Full-Batch ohne Momentum bleibt auf dem
Plateau J ≈ 0,125 (nur ¼ der Updates), mit Momentum 0,9 nach 11 941.

### E-25: Experimente mit vielen Netzen gleichzeitig (`experimente/parallel.py`)

**Entscheidung:** N Netze gleicher Struktur als Matrix-Stapel (N, Neuronen, Eingänge), jeder
Schritt per `einsum` für alle zugleich; Startgewichte aus `nn.NeuralNetwork` mit dem Seed.

**Begründung:** Die Studien brauchen > 1 000 Läufe; mit `nn/` wären das Stunden. Kein
Ersatz für `nn/` (die Angabe verlangt die Klassen Neuron/Layer), nur ein Messwerkzeug.

**Nachweis:** Abweichung zu `nn/` ≤ 9·10⁻¹⁶ (sigmoid/tanh/ReLU), und die Messung aus E-11 wird
**exakt** reproduziert (86 richtig, 58 sauber) — die damaligen Seeds waren 0–99.

### E-26: Seed-Studie — lokale Minima sind ein Problem der Netzgröße 2-2-1

| 100 Seeds | 2-2-1 klein | 2-2-1 Xavier | 2-3-1 klein | 2-3-1 Xavier | 2-4-1 klein | 2-4-1 Xavier |
|---|---|---|---|---|---|---|
| richtig nach 20 000 | 86 | 75 | 96 | 99 | 99 | 100 |
| hängt nach 80 000 noch | 7 | 24 | 0 | 0 | 0 | 0 |
| Median bis J < 0,01 | 15 919 | 5 790 | 16 220 | 4 960 | 15 800 | 4 800 |

**Folgerung:** E-11 gilt nur für die Größe aus der Angabe. Ab 3 Hidden-Neuronen gibt es keine
dauerhaften Fehlschläge mehr, und Xavier ist dann gleich zuverlässig und ~3× schneller.
**Standard bleibt `klein`**, weil die Angabe 2-2-1 vorschreibt.

### E-27: Aktivierungen, Rauschen, Vektorisierung — Messergebnisse

- **Aktivierungen** (Ausgabe immer Sigmoid, weil Ziele 0/1): 7-Segment lernt mit tanh
  (Median 578–650 Epochen) und ReLU (413–474) 4–6× schneller als Sigmoid (2 466–2 680);
  ReLU verliert 1–2 von 50 Läufen. XOR 2-2-1 scheitert mit tanh/ReLU meistens (9–49 bzw.
  18–19 von 100): zu wenig Reserve, "tote" ReLU-Neuronen. Sigmoid bleibt Standard (Angabe).
- **Rauschen** (Eingang = Bit + N(0, σ)): Modell aus `models/7seg.json` bei σ = 0,2:
  78,5 % der Ziffern komplett richtig; mit σ = 0,2 trainiert: 95,6 %. **Gekippte Bits** im
  Training: bis p = 0,1 kein Verlust, bei p = 0,2 nur noch 77 %. Grund: Bei 4 Bit ergibt jedes
  gekippte Bit eine andere gültige Zahl — die Trainingsbeispiele widersprechen sich, echte
  Robustheit gegen Bitfehler ist mit dieser Kodierung unmöglich.
- **Vektorisiert** (`experimente/vektorisiert.py`): 4-16-7 3,8× schneller, 4-64-7 9,7×,
  XOR 2-2-1 aber 0,6× (numpy-Overhead bei 2×2-Matrizen). `nn/` bleibt objektorientiert, weil
  die Angabe die Klassenstruktur verlangt und sie den Code nachvollziehbar macht.

## 2026-10-19 — Überarbeitung der Visualisierung

### E-28: Seitentexte radikal gekürzt — Begründungen gehören in die Doku, nicht auf die Seite

**Entscheidung:** Jede Kapiteleinleitung in `vorlage.html` ist höchstens ein Satz: was man sieht
und was man anklicken kann. Hinweistexte aus dem JavaScript nennen nur noch das Ergebnis, nicht
die Herleitung. Zahlen, Formeln, Tabellen und Tooltips bleiben vollständig.

**Begründung:** Vorgabe des Users: zu viel unnötige Erklärung. Die Seite ist zum Ausprobieren
und Vorführen da. Die ausführlichen Begründungen (z. B. warum TF Standard langsamer ist, warum
Abweichungen bei 10⁻¹⁶ liegen) stehen bereits in `CLAUDE.md` (E-11, E-21, E-26, E-27) und
`README.md` und gehen damit nicht verloren.

**Verworfen:** Lange Texte einklappbar machen ("Mehr erfahren") — mehr UI-Elemente für Inhalte,
die schon in der Doku stehen.

## 2026-10-19 — Phase 9: Ziffern zeichnen und erkennen

### E-29: Erweiterung "Ziffer zeichnen" — Wahl des Users aus vier Vorschlägen

**Entscheidung:** Neues Kapitel auf der Seite: Ziffer mit Maus/Finger malen, ein zweites Netz
(64 → 48 → 10) erkennt sie live. Vorgeschlagen waren außerdem eine 3D-Kostenlandschaft, der
C-Code als WebAssembly und Adversarial-Beispiele; der User wählte das Zeichnen.

### E-30: Softmax als Aktivierung der Schicht, nicht des Neurons

**Entscheidung:** Neurons einer Softmax-Schicht rechnen nur z (Registry-Eintrag `"softmax"` =
Identität). `Layer.forward` wendet `softmax` auf den Vektor an und trägt jedem Neuron sein a nach;
`Layer.backward` benutzt die volle Jacobi-Matrix `dJ/dz_i = a_i · (dJ/da_i − Σ_j dJ/da_j·a_j)`.

**Begründung:** Softmax braucht alle z der Schicht, ein einzelnes Neuron kennt nur sein eigenes.
Die Neuron/Layer-Struktur der Angabe bleibt so erhalten, der Name `softmax` steht wie jede andere
Aktivierung in der JSON. Bestehende Pfade sind unverändert: Der Bit-Gleichheitstest mit dem
Ausgangsstand (E-13) bleibt grün.

**Verworfen:** Softmax als eigene Klasse `SoftmaxLayer` — zweite Layer-Art für Export, Laden und
Visualisierung; die Fallunterscheidung an zwei Stellen in `Layer` ist kleiner.

### E-31: Cross-Entropy mit direktem delta = A − Y

**Entscheidung:** `NeuralNetwork(loss="cross_entropy")`. Bei Cross-Entropy übergibt `backward` der
Output-Schicht `A − Y` direkt als delta (`errors_are_deltas`), statt `dJ/dA = −Y/A` durch die
Softmax-Ableitung zu schicken. Erlaubt nur mit softmax- oder sigmoid-Ausgang (für beide gilt die
Vereinfachung). `"loss"` steht nur in der JSON, wenn es nicht `"mse"` ist.

**Begründung:** Mathematisch identisch (Herleitung im Kommentar von `backward`), aber ohne Division
durch a, die bei a ≈ 0 überlaufen würde. MSE bleibt Standard, damit XOR/7-Segment und ihre
Exportdateien exakt wie vorher sind. **Nachweis:** `tests/test_softmax.py`, Gradient-Check für
softmax+CE, softmax+MSE (volle Jacobi-Matrix), sigmoid+CE und zwei Hidden-Schichten: < 10⁻⁶.

### E-32: Datensatz und Netzgröße — gemessen

**Entscheidung:** UCI-Ziffern 8×8 (1797 Bilder, über `sklearn.datasets.load_digits` — sklearn
liefert nur die Daten), 20 % fest als Testmenge. 64 → 48 (Sigmoid) → 10 (Softmax), Xavier,
α = 0,05, 80 Epochen, gemischt, jede Epoche zufällig um ±1 Pixel verschoben.

**Messung** (Testgenauigkeit der letzten 10 Epochen): 32 Hidden/α 0,1/40 Ep.: 91,9–95,3 %, stark
schwankend; 32/0,05/80: 91,6–95,0 %; 32/0,02/120: 94,2–96,1 %; **48/0,05/80: 95,8–96,7 %**.
Ergebnis: **96,7 % Test** (347/359), 97,4 % Training, 47 s.

**Einschränkung, ehrlich:** Die Größe wurde anhand der Testmenge gewählt; eine eigene
Validierungsmenge gibt es nicht. Bei drei Kandidaten ist der Effekt klein, die 96,7 % sind aber
nicht streng unabhängig.

**Verworfen:** MNIST (28×28, 60 000 Bilder): mit dem Neuron-für-Neuron-Code Stunden pro Training.

### E-33: Zeichnung wie die Trainingsbilder aufbereiten

**Entscheidung:** Im Browser werden die Striche gespeichert und neu auf 32×32 gerendert: Höhe auf
30 px normiert (Breite ≤ 26), zentriert, dann je 4×4 Pixel gemittelt → 8×8. Die Strichstärke wird
so gewählt, dass die Tintenmenge der des Datensatzes entspricht (19,5 von 64 Feldern, gemessen; bei
allen Ziffern fast gleich).

**Begründung:** Der Datensatz ist so normiert (Ziffern füllen die Höhe, gleiche Tintenmenge).
Ohne diese Schritte hinge das Ergebnis davon ab, wie groß und wie dick man malt.

**Geprüft** (Headless-Edge, synthetische Striche): 0–9 alle richtig erkannt. Eine 7 **ohne**
Querstrich wird als 3 erkannt — die 7er im Datensatz haben fast alle einen (europäische
Schreibweise). Hinweis auf der Seite: "7 mit Querstrich". Browser ↔ Python: 3,3·10⁻¹⁶.

### E-34: Prüfung gegen die Angabe — nur Part 6 offen, bleibt so

**Befund (19.10.2026):** Part 1–5, alle Bonus-Aufgaben und optionalen Erweiterungen erfüllt und
nachgeprüft (Tabellen in `datasets.py` = PDF, Tests, C-Programme). Part 6 (ESP32) fehlt.
**Entscheidung des Users:** ESP32 wird nicht gemacht; es bleibt bei der Visualisierung (E-01).

## 2026-10-19 — Neugestaltung der Visualisierung

### E-35: Fünf Bereiche als Tabs, immer nur ein Kapitel sichtbar

**Befund:** Vorgabe des Users: Die Seite ist "sehr unübersichtlich". Ursache war die Menge, nicht
die Farbe: 12 Kapitel untereinander, ein großer Kopfbereich, alle Regler gleichzeitig offen.

**Entscheidung:**
- Kopfzeile mit fünf Bereichen: **Decoder · Zeichnen · So rechnet es · Training · Ergebnisse**.
  Bereiche mit mehreren Kapiteln haben eine zweite Leiste; es ist immer genau ein Kapitel sichtbar.
  Die Adresse merkt sich das Kapitel (`#neuron`, `#live`), alte Links funktionieren weiter.
- "So rechnet es" hat eine Eingabe-Leiste (0–F), damit man die Ziffer wechseln kann, ohne zum
  Decoder zurückzuspringen.
- Werkbank: Feinregler und Rauschen aufklappbar; die Übersicht der 16 Eingaben gehört zum Decoder.
  Live-Training: Einstellungen aufklappbar. Kennzahlen aus dem Kopf wandern zu "Ergebnisse".
- Ruhigere Optik: kompakte Kopfzeile statt Titelblock, keine Eyebrow-Zeilen mehr, mehr Abstand,
  weichere Karten. Farben bleiben: LED-Orange = Aktivierung, Teal/Magenta = Vorzeichen.
- Live-Training stoppt, wenn man das Kapitel verlässt (sonst rechnet es unsichtbar weiter).

**Begründung:** Die Rechenlogik und alle Diagramme bleiben unverändert; geändert ist nur, wann
was sichtbar ist. Versteckte Kapitel werden trotzdem beim Start gezeichnet (SVG mit viewBox,
Canvas mit fester Pixelzahl brauchen keine sichtbare Größe).

**Geprüft:** Headless-Edge, alle 12 Kapitel per Adresse geöffnet und fotografiert (hell, dunkel,
Handybreite), keine JS-Fehler, Nachweis-Tabelle 7/7 erfüllt. Gefunden und behoben: Beim Öffnen
mit `#kapitel` sprang der Browser zum Anker und verdeckte die Kapitel-Leiste
(`scroll-behavior: smooth` entfernt, nach dem Laden nach oben).

**Verworfen:** Nur Farben ändern (löst die Menge nicht); eine Startseite mit Kacheln vor dem
Decoder (ein Klick mehr bis zum eigentlichen Netz).

## 2026-10-19 — Ziffernerkennung auf MNIST umgestellt

### E-36: MNIST 14×14 statt UCI 8×8 — ersetzt E-32 und E-33

**Befund:** Vom User gemeldet: Zeichnen funktioniert "noch nicht 100 %". Zwei Ursachen: Der
UCI-Datensatz ist klein (1438 Trainingsbilder, wenige Dutzend Schreiber), und seine Aufbereitung
musste im Browser geschätzt werden (E-33).

**Entscheidung:** MNIST (60 000 Trainings-, 10 000 Testbilder, ~500 Schreiber), mit Zustimmung
des Users heruntergeladen (`mnist.npz`, 11 MB, Keras-Spiegel, im Keras-Cache). Auflösung 14×14
(je 2×2 Pixel gemittelt, Wahl des Users), Netz 196 → 64 (Sigmoid) → 10 (Softmax), Cross-Entropy,
α = 0,05, 15 Epochen, gemischt, ohne Augmentation. Der feste MNIST-Testsatz ersetzt die eigene
20-%-Teilung — damit ist auch die Einschränkung aus E-32 (Auswahl anhand der Testmenge) erledigt.

**Messungen vorab:** MNIST-Ziffern sind 20 px hoch (Median), Schwerpunkt bei (14,00; 13,99).
Die Tintenmenge hängt von der Ziffer ab (1: 60, 0: 136 Pixel bei 28×28) — anders als bei UCI ist
also die **Strichstärke** konstant, nicht die Tinte. Training mit `nn/`: 0,27 ms pro Bild,
~19 s pro Epoche.

**Browser-Aufbereitung jetzt genau nach dem MNIST-Verfahren:** längere Seite auf 20 px, Strich
2,7 px breit auf 28×28, Schwerpunkt auf (14, 14) verschoben, dann 2×2 gemittelt → 14×14.

**Verworfen:** 28×28 (Wahl des Users: 14×14; ~4× langsamer, kaum besser); EMNIST (noch größer,
für Ziffern kein Vorteil).

**Ergebnis:** **97,70 % auf den 10 000 MNIST-Testbildern** (99,04 % Training), 314 s.
Häufigste Verwechslung: 4 → 9 (18×). Synthetische Zeichnungen (Headless-Edge, dieselben 11 wie
in E-33): alle richtig, meist 97–100 % sicher (vorher teils 39–80 %); auch die 7 ohne Querstrich,
die vorher als 3 erkannt wurde. Klein in eine Ecke gezeichnet: identisches Ergebnis.
Browser ↔ Python: 4,4·10⁻¹⁶. Die Seite wächst durch 500 eingebettete Testbilder auf 1,1 MB.
Bleibt offen: Echte Mauszeichnungen sind wackeliger als die synthetischen Striche; das lässt
sich nur von Hand prüfen.

## 2026-10-19 — Abgabe-Dokumentation

### E-37: Quarto, ein Dokument für HTML und PDF, Bilder per Skript

**Entscheidung:** `dokumentation/dokumentation.qmd`, gerendert als HTML und als PDF über Typst
(mit Quarto 1.10 geliefert). Autor: Anes Sejdic (Angabe des Users). Kein ausgeführter Python-Code
im Dokument: Zahlen stammen aus den Ergebnisdateien und diesem Protokoll, Bilder aus zwei Skripten
(`screenshots.py`: Edge headless, helles Schema, pro Kapitel optional ein Skript vor dem Foto;
`abbildungen.py`: matplotlib).

**Begründung:** Quarto mit Jupyter bräuchte `QUARTO_PYTHON`, weil `python` auf Python 3.12 ohne
numpy zeigt — ein Stolperstein beim Neu-Rendern. Typst statt LaTeX: kein weiterer Download.
Helles Schema, weil die PDF gedruckt wird.

**Gefunden und behoben beim Prüfen der gerenderten Seiten:** Typst liest keine Dateien außerhalb
des Ordners (Diagramme werden nach `bilder/` kopiert); `\!` und `0{,}1` in Formeln werden falsch
umgesetzt (entfernt bzw. Zahl aus der Formel genommen); deutsche Anführungszeichen erschienen als
„…„ (gerade Anführungszeichen, Typst setzt sie selbst); MathJax bräuchte Internet (MathML);
doppelter Format-Link (abgeschaltet, Hinweis auf das PDF nur in der HTML-Fassung).

**Verworfen:** LaTeX/TinyTeX (großer Download), Quarto-Buchprojekt mit mehreren Dateien (für
26 Seiten unnötig), dunkle Screenshots (schlecht zu drucken).

## 2026-10-19 — Git-Repo für die Abgabe

### E-38: Öffentliches GitHub-Repo, ohne Angabe, venv und Build-Ergebnisse

**Entscheidung (User):** öffentliches Repo auf GitHub, Abgabe über den Link. Die Angabe
`4AHITS_NN_from_scratch.pdf` (Unterlage der Lehrkraft) kommt nicht hinein.

**Nicht im Repo (`.gitignore`):** `.venv-tf/` (1,6 GB, nach README neu anlegbar), `*.exe`/`*.o`
(mit `mingw32-make` neu bauen), `__pycache__`, `.pytest_cache`, `dokumentation/.quarto/`, die Angabe.
MNIST liegt ohnehin außerhalb des Projekts (Keras-Cache, Download-Anleitung im README).
**Im Repo:** auch die erzeugten Dateien `visualisierung/index.html`, `dokumentation/*.html|.pdf`
und `models/` — die Lehrkraft soll Ergebnisse ansehen können, ohne etwas zu installieren.

**Vor dem Veröffentlichen geprüft:** keine Benutzerpfade, E-Mail-Adressen oder Zugangsdaten in den
93 Dateien (Treffer nur in eingebettetem Bibliotheks-Code). Sichtbar bleiben Name und Schul-Adresse
in den Commit-Metadaten (Git-Konfiguration des Users). `.gitattributes`: Text mit LF, Bilder/PDF binär.

---

## Offene Punkte / noch zu entscheiden

- ~~Hidden-Layer-Größe im TF-Vergleich~~ → erledigt, siehe E-20.
- ~~Versionskontrolle~~ → erledigt, siehe E-38.
