# Prompts.md — Protokoll aller Prompts

Jeder Prompt an Claude wird hier **gekürzt** festgehalten: was verlangt wurde und was dabei
herauskam. Die ausführliche Begründung der Entscheidungen steht in [CLAUDE.md](CLAUDE.md),
dieses Dokument ist nur der rote Faden der Zusammenarbeit.

Regel dazu: R-02 in [CLAUDE.md](CLAUDE.md).

---

## P-01 · 29.09.2026 · Bestandsaufnahme und Implementierungsplan

**Prompt (gekürzt):** Die Aufgabe steht in `4AHITS_NN_from_scratch.pdf`. Den bereits fertigen
Teil ansehen und zuerst eine Übersicht über den Stand erstellen (`Ausgangsstand.md`), danach einen
Implementierungsplan (`Implementation.md`). Parallel ein `CLAUDE.md` mitführen, das alle wichtigen
Entscheidungen und Gedankenschritte protokolliert. **Planänderung:** kein ESP32 (Part 6), sondern
eine Visualisierung des 7-Segment-Decoders. Zuerst die Angabe vollständig erfüllen, danach
Erweiterungen, die die Arbeit von gut auf exzellent heben.

**Rückfragen von Claude und Antworten:**
| Frage | Antwort |
|---|---|
| TensorFlow läuft nicht unter Python 3.14 — wie weiter? | Python 3.12 + eigenes venv |
| Kein C-Compiler installiert — wie weiter? | MinGW-w64 installieren |
| Form der Visualisierung? | interaktive HTML-Seite |
| Code-Struktur? | modulares Paket + Skripte |

**Ergebnis:** `Ausgangsstand.md`, `Implementation.md` (Phasen 0–8), `CLAUDE.md` (E-01 bis E-07).

---

## P-02 · 29.09.2026 · Kommentar-Regel

**Prompt (gekürzt):** In `CLAUDE.md` festhalten, dass immer Kommentare geschrieben werden, sodass
ein Mensch den Code im Nachhinein versteht — und das gleich im Ausgangszustand umsetzen.

**Ergebnis:** Regel R-01 in `CLAUDE.md`. Die Kommentierung selbst folgte mit P-03.

---

## P-03 · 29.09.2026 · Ausgangszustand kommentieren

**Prompt (gekürzt):** Ja, jetzt die Kommentare machen.

**Ergebnis:** `neural_network.py` vollständig kommentiert (175 → 489 Zeilen). Logik nachweislich
unverändert: Syntaxbaum ohne Docstrings identisch, Programmausgabe identisch.
Zeilenverweise in `Ausgangsstand.md` angepasst, dort Abschnitt 7 ergänzt.

---

## P-04 · 29.09.2026 · Phase 1 und Prompt-Protokoll

**Prompt (gekürzt):** Weiter im Plan mit Phase 1. Zusätzlich ein `Prompts.md` anlegen, in dem alle
Prompts gekürzt protokolliert werden.

**Ergebnis:**
- `Prompts.md` angelegt, Regel R-02 in `CLAUDE.md`.
- Paket `nn/` (`activations`, `neuron`, `layer`, `network`) und `datasets.py`;
  `neural_network.py` ist jetzt ein schlanker Einstiegspunkt, das Original liegt unverändert
  in `archiv/`.
- `tests/test_kern.py`: 28 Tests grün. Der Gradient-Check stimmt auf 7·10⁻¹¹, und der
  Lauf ist bit-identisch mit dem Ausgangsstand.
- **Planänderung nach Messung:** Xavier wurde nicht Standard-Initialisierung. Bei XOR
  bleibt es in 24 von 100 Läufen dauerhaft hängen, `0.1` nur in 7 von 100.
  Schwachstelle 5 in `Ausgangsstand.md` wurde als widerlegt markiert.
- Entscheidungen E-08 bis E-13 in `CLAUDE.md`.

---

## P-05 · 12.10.2026 · Plan fertigstellen und ausführliche Visualisierung

**Prompt (gekürzt):** Das neuronale Netz fertig machen: im Plan nachsehen, was noch offen ist, und
alles abarbeiten. Am Ende unbedingt eine sehr ausführliche, detaillierte Visualisierung des Netzes
(z. B. HTML-Seite). Vorher alle Dateien wie `CLAUDE.md` ansehen.

**Rückfragen:** keine — die Installationen (Python 3.12, MinGW-w64) waren in P-01 bereits geklärt.

**Ergebnis:** Phasen 2–8 abgeschlossen, Entscheidungen E-14 bis E-27 in `CLAUDE.md`.
- Phase 2: `nn/persistence.py` (JSON, TXT, Header) — bit-exakt dank `repr`.
- Phase 3: `train_xor.py`, `test_xor_load.py`, `train_7seg.py`, `test_7seg_load.py`, `auswertung.py`;
  XOR 4/4, 7-Segment 70/70, Abweichung nach dem Laden 0.
- Phase 4: `compare_tensorflow.py` (TF 2.21 in `.venv-tf`); TF mit gleichen Startgewichten = eigene
  Implementierung bis 10⁻¹⁶.
- Phase 5/6: `c_forward/`, `c_header_init/`; 70/70, Abweichung 2·10⁻¹⁶.
- Phase 7: `visualisierung/index.html` mit 11 Kapiteln (Werkbank, Rechenweg, Inspektor, Matrizen,
  alle Ziffern, Backprop, Training/TF, Live-Training, XOR, Experimente, Nachweis).
- Phase 8: Mini-Batch/Momentum, 55 Tests, Seed-Studie, Aktivierungsvergleich, Rauschen, Benchmark,
  Plots, `README.md`.

---

## P-06 · 19.10.2026 · Stand ansehen

**Prompt (gekürzt):** Schau dir an, was du bis jetzt findest.

**Rückfragen:** keine.

**Ergebnis:** Bestandsaufnahme ohne Code-Änderung. Alle Phasen 0–8 laut Doku abgeschlossen und
nachgeprüft: `py -3.14 -m pytest` → 55/55 grün (inkl. C-Abgleich, gcc vorhanden),
`test_xor_load.py` 4/4 und `test_7seg_load.py` 70/70 mit Abweichung 0, TensorFlow im `.venv-tf`
vorhanden. **Befund:** `python` zeigt inzwischen auf Python 3.12 (ohne numpy/pytest); die Befehle
aus `README.md` laufen nur mit `py -3.14`.

---

## P-07 · 19.10.2026 · Befehle auf `py -3.14` umstellen

**Prompt (gekürzt):** Ja, die Befehle in `README.md` und `Implementation.md` auf `py -3.14` umstellen.

**Ergebnis:** 17 Aufrufe `python …` → `py -3.14 …` (README 11, Plan 6), dazu ein Hinweis bei den
Voraussetzungen in der README. Unverändert: `.venv-tf\Scripts\python` (TensorFlow-venv) und
`py -3.12 -m venv`.

---

## P-08 · 19.10.2026 · HTML-Seite starten

**Prompt (gekürzt):** Starte die HTML-Seite.

**Ergebnis:** `visualisierung/index.html` im Standardbrowser geöffnet (der Browser-Bereich der App
öffnet keine lokalen Dateien).

---

## P-09 · 19.10.2026 · Texte der Visualisierung kürzen, Erweiterungsideen

**Prompt (gekürzt):** Zu viele (unnötige) Erklärungen auf der HTML-Seite. Kürzen, sodass es
verständlich bleibt, und überlegen, welche Erweiterungen (Features, Tools) noch möglich wären.

**Ergebnis:** In `vorlage.html` alle 11 Kapiteleinleitungen auf einen Satz gekürzt, dazu ~20
Hinweistexte aus dem JavaScript. Rechenlogik unverändert; `index.html` neu erzeugt, lädt ohne
JS-Fehler. Erweiterungsideen im Chat vorgeschlagen. Entscheidung: E-28 in `CLAUDE.md`.

---

## P-10 · 19.10.2026 · "Lieber was Cooleres" — Ziffern zeichnen

**Prompt (gekürzt):** Statt Hex-Decoder oder Neuronen-Ausschalten lieber etwas Cooleres.

**Rückfrage von Claude und Antwort:**
| Frage | Antwort |
|---|---|
| Welche Erweiterung? (Ziffern zeichnen / 3D-Kostenlandschaft / C im Browser / Netz austricksen) | Ziffern zeichnen |

**Ergebnis:** Phase 9. Softmax und Cross-Entropy in `nn/`, `tests/test_softmax.py` (70 Tests grün),
`train_digits.py` (64-48-10, 96,7 % auf 359 Testbildern), neues Kapitel "Zeichnen" auf der Seite.
Entscheidungen E-29 bis E-33.

---

## P-11 · 19.10.2026 · Angabe prüfen, nur berichten

**Prompt (gekürzt):** Im PDF nachsehen, ob alle Punkte erfüllt sind. Berichten, noch nichts ändern.

**Ergebnis:** Bericht im Chat, keine Änderung. Part 1–5 sowie alle Bonus- und optionalen Aufgaben
erfüllt und nachgeprüft; Part 6 (ESP32) fehlt (Planänderung E-01). E-34.

---

## P-12 · 19.10.2026 · Seite neu gestalten

**Prompt (gekürzt):** ESP32 wird nicht gemacht. Die Webseite komplett neu gestalten, sie ist viel zu
unübersichtlich; beste Strategie selbst überlegen (Farben, Anordnung, Erklärungen).

**Ergebnis:** Phase 10. Fünf Bereiche als Tabs, immer nur ein Kapitel sichtbar, Eingabe-Leiste,
aufklappbare Zusatzregler, ruhigere Optik; Rechenlogik unverändert. Alle 12 Kapitel per Screenshot
geprüft (hell, dunkel, Handy), keine JS-Fehler. E-35.

---

## P-13 · 19.10.2026 · Größeres Datenset fürs Zeichnen

**Prompt (gekürzt):** Das Zeichnen funktioniert noch nicht 100 %. Kann man ein größeres Datenset nehmen?

**Rückfragen von Claude und Antworten:**
| Frage | Antwort |
|---|---|
| MNIST herunterladen (mnist.npz, 11 MB, Keras-Spiegel)? | ja |
| Auflösung 14×14 oder 28×28? | 14×14 |

**Ergebnis:** `train_digits.py` auf MNIST umgestellt (196-64-10, 97,7 % auf 10 000 Testbildern),
Aufbereitung im Browser nach dem MNIST-Verfahren. Alle 11 synthetischen Testzeichnungen richtig.
Zwischendurch `vorlage.html` durch eine fehlerhafte Ersetzung beschädigt und aus dem letzten
Export (`index.html`) vollständig wiederhergestellt. E-36.

---

## P-14 · 19.10.2026 · Dokumentation mit Quarto

**Prompt (gekürzt):** Am Ende eine Dokumentation mit Quarto, mit Screenshots, Formeln und allem,
was dazugehört.

**Rückfragen von Claude und Antworten:**
| Frage | Antwort |
|---|---|
| Quarto über winget installieren (Posit.Quarto, ~130 MB)? | ja |
| Format? | HTML und PDF |
| Autor? | Anes Sejdic |

**Ergebnis:** Quarto 1.10.19 installiert. `dokumentation/dokumentation.qmd` → `dokumentation.html`
(5,7 MB, eigenständig) und `dokumentation.pdf` (26 Seiten, Typst) mit Theorie und Formeln,
Code-Auszügen, Exportformaten, Ergebnistabellen, 12 Screenshots und 9 Diagrammen.
Bilder per `screenshots.py` und `abbildungen.py` reproduzierbar. Alle Seiten gerendert geprüft,
Darstellungsfehler (Formeln, Anführungszeichen, Bildpfade) behoben. E-37.

---

## P-15 · 19.10.2026 · Git-Repo für die Abgabe

**Prompt (gekürzt):** Alles in ein Git-Repo, das beim Lehrer abgegeben wird.

**Rückfragen von Claude und Antworten:**
| Frage | Antwort |
|---|---|
| Plattform? | GitHub |
| Sichtbarkeit? | öffentlich |
| Angabe-PDF ins Repo? | nein |

**Ergebnis:** `.gitignore` und `.gitattributes`, `git init -b main`, erster Commit mit 93 Dateien (15 MB),
vorher auf private Daten geprüft. E-38.

---

## P-16 · 19.10.2026 · Claude aus den Contributors entfernen

**Prompt (gekürzt):** Kann man "Claude" aus den Contributors wegmachen?

**Ergebnis:** Co-Author-Zeile aus beiden Commits entfernt, Dateibaum unverändert, mit
`--force-with-lease` neu hochgeladen. Künftige Commits ohne Claude-Zeile. E-39.

---

## P-17 · 19.10.2026 · GitHub Pages für Visualisierung und Dokumentation

**Prompt (gekürzt):** Zwei GitHub Pages einrichten: Visualisierung und Quarto-Dokumentation.

**Ergebnis:** Eine Pages-Seite mit Startseite `index.html` und zwei Unterseiten, `.nojekyll`, Links in
der README. Adresse: https://anes7374.github.io/5AHITS_INSY/ E-40.

