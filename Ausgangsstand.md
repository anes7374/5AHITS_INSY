# Ausgangsstand

Bestandsaufnahme vom **29.09.2026**, vor Beginn des strukturierten Ausbaus.

Grundlage: die Angabe `4AHITS_NN_from_scratch.pdf` (Parts 1–6) und die bestehende
Implementierung in `neural_network.py` — eine einzelne Datei, ursprünglich 175 Zeilen.
Sie wurde danach vollständig kommentiert (siehe Abschnitt 7) und mit Phase 1 unverändert nach
[archiv/neural_network_ausgangsstand.py](archiv/neural_network_ausgangsstand.py) verschoben.
**Alle Zeilenverweise beziehen sich auf diese archivierte, kommentierte Fassung**; ihre Logik
ist identisch mit dem Original.

---

## 1. Kurzfassung

Der **fachliche Kern steht und ist korrekt**: Neuron, Layer und Netz sind sauber getrennt,
der Forward-Pass rechnet richtig, die Backpropagation folgt exakt den Formeln aus dem
Theorieteil der Angabe, und XOR wird erfolgreich gelernt. Damit sind Part 1 Steps 1–3
abgeschlossen und Part 2 Step 1 zum größten Teil.

Was fehlt, ist alles rund um **Persistenz und Portierung** — also genau die Schritte, auf
denen die Parts 2 bis 5 aufbauen: JSON-, TXT- und Header-Export, der 7-Segment-Decoder,
der TensorFlow-Vergleich und die beiden C-Implementierungen.

Grob geschätzter Fertigstellungsgrad der Angabe: **ca. 25 %** — aber es ist das Fundament,
auf dem der Rest steht, nicht die Fassade.

---

## 2. Fortschritt je Angabe-Schritt

| Part | Schritt | Status | Anmerkung |
|---|---|:---:|---|
| **1** | Step 1 — Klassen `Neuron`, `Layer`, `NeuralNetwork` | ✅ | vollständig, alle geforderten Felder vorhanden |
| **1** | Step 2 — `forward` / `predict`, Sigmoid | ✅ | inkl. Caching für den Backward-Pass |
| **1** | Step 3 — `backward` / `train` | ✅ | SGD, Kettenregel korrekt, Kostenhistorie |
| **1** | Step 4 — JSON Export / Import | ❌ | fehlt vollständig |
| **1** | Step 5 — TXT-Export (menschenlesbar) | ❌ | fehlt vollständig |
| **1** | Step 6 — Header-Datei `.h` für C | ❌ | fehlt vollständig |
| **2** | Step 1 — XOR trainieren (2-2-1, 20000, α=0.1) | ⚠️ | Training läuft und konvergiert; der geforderte **JSON-Export fehlt** |
| **2** | Step 2 — Netz laden und testen | ❌ | setzt Step 4 aus Part 1 voraus |
| **2** | Step 3 — 7-Segment-Decoder (4-16-7) | ❌ | Datensatz und Training fehlen |
| **2** | Step 4 — 7-Segment laden und testen | ❌ | fehlt |
| **3** | TensorFlow-Vergleich | ❌ | fehlt; TF ist auf dem System nicht lauffähig (siehe Abschnitt 5) |
| **4** | Forward-Only in C, Init aus `.txt` | ❌ | fehlt |
| **5** | Forward-Only in C, Init aus `.h` | ❌ | fehlt |
| **6** | ESP32 + physisches Display | ⊘ | **entfällt laut Planänderung**, ersetzt durch interaktive Visualisierung |

Legende: ✅ fertig · ⚠️ teilweise · ❌ offen · ⊘ gestrichen

---

## 3. Was inhaltlich bereits sitzt

### Architektur

Die Dreiteilung aus der Angabe ist exakt umgesetzt:

- **`Neuron`** ([neural_network_ausgangsstand.py:102-188](archiv/neural_network_ausgangsstand.py#L102-L188)) hält `weights` und `bias`,
  besitzt eine austauschbare Aktivierungsfunktion, rechnet in `activate()` das Skalarprodukt
  plus Bias und legt `_last_inputs`, `_last_raw` und `_last_output` für den Backward-Pass ab.
  `update()` wendet die Gradienten an.
- **`Layer`** ([neural_network_ausgangsstand.py:195-273](archiv/neural_network_ausgangsstand.py#L195-L273)) verwaltet die Neuronenliste,
  `forward()` sammelt die Ausgaben, `backward()` berechnet Deltas, aktualisiert die Neuronen
  und gibt den Fehler für die vorgelagerte Schicht zurück.
- **`NeuralNetwork`** ([neural_network_ausgangsstand.py:280-418](archiv/neural_network_ausgangsstand.py#L280-L418)) baut die Layer aus
  einer Größenliste wie `[2, 2, 1]`, bietet `forward`, `predict`, `backward` und einen
  `train`-Loop mit Kostenhistorie.

### Mathematik

Die Umsetzung deckt sich mit dem Theorieteil der Angabe:

| Formel aus der Angabe | Fundstelle im Code |
|---|---|
| Kostenfunktion `J = ½·(Ŷ − Y)²` | [neural_network_ausgangsstand.py:405](archiv/neural_network_ausgangsstand.py#L405) |
| Startfehler `∂J/∂A = A − Y` | [neural_network_ausgangsstand.py:365](archiv/neural_network_ausgangsstand.py#L365) |
| Delta `δ = ∂J/∂A · sigma_ableitung(z)` | [neural_network_ausgangsstand.py:251](archiv/neural_network_ausgangsstand.py#L251) |
| Gewichtsupdate `W ← W − α·δ·x` | [neural_network_ausgangsstand.py:186-188](archiv/neural_network_ausgangsstand.py#L186-L188) |
| SGD, ein Sample pro Iteration | [neural_network_ausgangsstand.py:398-407](archiv/neural_network_ausgangsstand.py#L398-L407) |

### Der entscheidende Detailpunkt — richtig gelöst

In [neural_network_ausgangsstand.py:254-273](archiv/neural_network_ausgangsstand.py#L254-L273) wird der Fehler für die vorherige
Schicht **vor** dem Gewichtsupdate berechnet, also noch aus den alten Gewichten (Auszug, gekürzt):

```python
# 2) Fehler für die vorherige Schicht: gewichtete Summe der Deltas.
#    WICHTIG: das muss mit den ALTEN Gewichten passieren, also vor dem Update,
#    sonst stimmt die Kettenregel nicht mehr.
prev_errors = np.zeros(num_inputs)
for neuron, delta in zip(self.neurons, deltas):
    prev_errors += delta * neuron.weights

# 3) Erst jetzt die Gewichte und Biases aktualisieren
```

Das ist der häufigste Fehler bei dieser Aufgabe: Wer zuerst updatet und dann rückpropagiert,
rechnet die Kettenregel mit bereits veränderten Gewichten und bekommt ein Netz, das nur noch
zufällig konvergiert. Hier ist es korrekt — und im Code auch begründet.

### Training

`train_xor()` ([neural_network_ausgangsstand.py:448-484](archiv/neural_network_ausgangsstand.py#L448-L484)) trainiert das
geforderte 2-2-1-Netz über 20000 Epochen mit α = 0.1 und gibt eine Ergebnistabelle mit
Rohausgabe, gerundetem Wert und Trefferzähler aus. Der fixe Seed `np.random.seed(1)` ist
bewusst gesetzt und im Code begründet — siehe dazu Schwachstelle 5 unten.

---

## 4. Schwachstellen und offene Baustellen im bestehenden Code

Keine davon ist ein Rechenfehler. Es sind Punkte, die beim Ausbau auf die Parts 2–5
zwangsläufig aufschlagen.

| # | Fundstelle | Befund | Warum es stört |
|---|---|---|---|
| **1** | `Neuron.__init__` ([:114-139](archiv/neural_network_ausgangsstand.py#L114-L139)) | Die Aktivierung wird nur als Funktionsobjekt gespeichert, ohne Namen | **Blocker für Part 1 Step 4**: ein Funktionsobjekt lässt sich nicht nach JSON serialisieren. Es braucht eine Registry, die den Namen `sigmoid` auf die Funktion abbildet. |
| **2** | `Layer.__init__` ([:204-215](archiv/neural_network_ausgangsstand.py#L204-L215)) | `num_neurons` und `num_inputs` werden nicht als Felder abgelegt; `backward` liest die Eingangszahl über `len(self.neurons[0].weights)` | Funktioniert, aber für Export und `.h`-Generierung braucht man die Struktur explizit. |
| **3** | `sigmoid_derivative` ([:77-95](archiv/neural_network_ausgangsstand.py#L77-L95)) | berechnet `sigmoid(x)` erneut, obwohl `_last_output` bereits gecacht ist | Doppelte `exp`-Aufrufe in jedem Backward-Schritt. Die Angabe schreibt die Ableitung ohnehin als `A·(1−A)` — also über die Ausgabe. |
| **4** | `NeuralNetwork.backward` ([:366-370](archiv/neural_network_ausgangsstand.py#L366-L370)) | propagiert den Fehler auch durch die erste Schicht zurück | Das Ergebnis wird verworfen — unnötige Rechenarbeit pro Sample. |
| **5** | Gewichts-Initialisierung ([:132-136](archiv/neural_network_ausgangsstand.py#L132-L136)) | `np.random.randn(n) * 0.1`, Bias = 0 | ~~Sehr kleine Startgewichte. XOR mit nur 2 Hidden-Neuronen bleibt dadurch je nach Seed in lokalen Minima hängen. Der Seed behebt die Ursache nicht.~~ **Durch Messung widerlegt (Phase 1):** Die kleinen Startgewichte sind bei XOR 2-2-1 nicht die Ursache, sondern die *zuverlässigere* Wahl: dauerhaft hängen bleiben 7 von 100 Läufen gegenüber 24 von 100 mit Xavier. Sie machen das Training nur langsamer. Die Fehlschläge kommen von der knappen Architektur mit nur 2 Hidden-Neuronen. Der feste Seed ist damit berechtigt: Er macht den Lauf reproduzierbar. Details: [CLAUDE.md](CLAUDE.md), E-11. **Keine Schwachstelle.** |
| **6** | Dateiaufbau ([:421-489](archiv/neural_network_ausgangsstand.py#L421-L489)) | XOR-Trainingsdaten und Trainingsskript stehen in derselben Datei wie die Bibliothek | Spätestens mit dem 7-Segment-Netz, dem TF-Vergleich und den C-Teilen wird das unübersichtlich. |
| **7** | — | Keine Tests, keine Eingabevalidierung, keine Accuracy-Metrik außer der XOR-Tabelle | Die Angabe nennt eine Testsuite als Bonus; für den Nachweis "Export/Import verlustfrei" braucht es ohnehin Vergleichsmetriken. |

Zusätzlich als Beobachtung, noch kein Problem: `activate()` legt pro Neuron und pro Sample ein
neues numpy-Array an. Beim 7-Segment-Netz sind das 20000 Epochen × 10 Samples × 23 Neuronen —
in der Größenordnung einer Minute Laufzeit. Tragbar, aber ein Kandidat für einen
Vektorisierungs-Benchmark als Erweiterung.

---

## 5. Umgebung

Vor der Planung geprüft, weil zwei Befunde den Ablauf direkt beeinflussen:

| Komponente | Stand | Konsequenz |
|---|---|---|
| Python | 3.14.2 | Kern läuft sofort |
| numpy | 2.4.4 | vorhanden |
| **TensorFlow** | **nicht installiert** | TF unterstützt Python 3.14 noch nicht (max. 3.12/3.13). **Part 3 ist ohne zusätzliche Installation nicht umsetzbar.** → Python 3.12 + eigenes venv |
| **C-Compiler** | **keiner vorhanden** | weder gcc, clang, MSVC noch MinGW. **Parts 4 und 5 sind ohne Installation nicht kompilierbar.** → MinGW-w64 |
| Versionskontrolle | kein Git-Repo | optional |

Beide Lücken sind reine Werkzeugfragen, keine inhaltlichen Hindernisse — sie müssen nur
eingeplant werden, bevor die entsprechenden Parts drankommen.

---

## 6. Fazit

Das Fundament ist tragfähig und in den kniffligen Punkten — Kettenregel, Reihenfolge von
Fehlerpropagation und Gewichtsupdate — korrekt. Der Ausbau ist damit kein Umbau, sondern
eine Erweiterung: Der bestehende Code wird in eine Paketstruktur überführt, um
Serialisierbarkeit ergänzt, und darauf setzen dann Persistenz, zweiter Datensatz,
TensorFlow-Vergleich, C-Portierungen und die Visualisierung auf.

Der konkrete Weg dorthin steht in [Implementation.md](Implementation.md).
Alle Entscheidungen werden fortlaufend in [CLAUDE.md](CLAUDE.md) protokolliert.

---

## 7. Nachtrag (29.09.2026): Kommentierung nach Regel R-01

Nach der Bestandsaufnahme wurde `neural_network.py` (heute
[archiv/neural_network_ausgangsstand.py](archiv/neural_network_ausgangsstand.py)) gemäß der neuen
Arbeitsregel R-01 (siehe [CLAUDE.md](CLAUDE.md)) vollständig kommentiert:

- **Dateikopf** mit Zweck, Bezug zur Angabe, Aufruf und einer **Übersetzungstabelle
  Code ↔ Angabe** (z. B. `_last_raw` = `Z[l]`, `delta` = `dJ/dZ[l]`). Das ist der wichtigste
  Zusatz: Die Angabe rechnet mit Matrizen pro Schicht, der Code Neuron für Neuron — die Tabelle
  verbindet beide Sichtweisen.
- **Docstring für jede Klasse und Funktion** mit Parametern, Längen der Vektoren,
  Rückgabewert und Seiteneffekten.
- **Formeln aus der Angabe** direkt an den Codezeilen, die sie umsetzen
  (Kettenregel in `Layer.backward`, Gradienten in `Neuron.update`).
- **`WICHTIG:`-Markierungen** an den drei Stellen, an denen eine naheliegende Änderung das
  Netz kaputt machen würde: Reihenfolge Fehlerberechnung/Update, `backward()` nur direkt nach
  `forward()` desselben Samples, Aktivierung und Ableitung müssen zusammenpassen.
- Verweise auf die Schwachstellen aus Abschnitt 4 direkt im Code, wo sie auftreten.

**Nachweis, dass nur Kommentare dazugekommen sind:**

| Prüfung | Ergebnis |
|---|---|
| Syntaxbaum (AST) vorher/nachher, Docstrings herausgerechnet | identisch |
| Programmausgabe vorher/nachher (fester Seed, 20000 Epochen) | identisch, 22 Zeilen, 4/4 korrekt |

Umfang: 175 → 489 Zeilen.
