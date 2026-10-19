"""
Phase 7 (Ersatz für Part 6 / ESP32): Daten für die interaktive Visualisierung exportieren.

Liest die trainierten Netze und schreibt sie zusammen mit Referenzwerten in die HTML-Seite:

    visualisierung/vorlage.html   die Seite mit Platzhalter  /*__NETZDATEN__*/null
            +  Daten (siehe unten)
    ->  visualisierung/index.html der fertige, eigenständige Einzeldatei-Export
        (per Doppelklick im Browser öffnen, keine Installation, kein Server nötig)

Warum die Daten in die Seite geschrieben werden statt sie zur Laufzeit zu laden: Browser
verbieten einer per Doppelklick geöffneten Seite (file://), andere Dateien per fetch() zu
lesen. Eine einzige Datei lässt sich außerdem einfach weitergeben.

WICHTIG -- es rechnet wirklich das trainierte Netz (CLAUDE.md, E-02): Exportiert werden nur
Gewichte und Biases. Den Forward-Pass rechnet die Seite in JavaScript selbst. Die Python-
Vorhersagen werden NUR zum Vergleich mitgeliefert (Abschnitt "Nachweis" der Seite).

Inhalt der Daten:
    netze.7seg / netze.xor   Modell (wie models/*.json), Trainingsprotokoll (Kostenverlauf
                             jede 10. Epoche), Trainingsdaten, Python-Referenzvorhersagen
    c                        Rohausgaben der C-Programme aus Part 4 und 5 (falls gebaut)
    tf                       Ergebnisse von compare_tensorflow.py (falls ausgeführt)
    experimente              Ergebnisse von experimente/*.py (Seed-Studie, Aktivierungen,
                             Rauschen, Benchmark), falls ausgeführt
    ziffern                  Phase 9: Ziffernerkennung aus train_digits.py (Netz, Testbilder,
                             Testgenauigkeit), falls trainiert

Voraussetzung: python train_xor.py und python train_7seg.py (optional: C-Programme bauen,
               compare_tensorflow.py ausführen -- dann zeigt die Seite auch diese Vergleiche)
Ausführen:     python visualisierung/export_daten.py
"""

import json
import subprocess
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

HIER = Path(__file__).resolve().parent
PROJEKT = HIER.parent
sys.path.insert(0, str(PROJEKT))   # damit nn/, datasets.py und auswertung.py gefunden werden

from auswertung import MODELS, lade_protokoll  # noqa: E402
from datasets import SEGMENT_NAMES  # noqa: E402
from nn import NeuralNetwork  # noqa: E402

PLATZHALTER = "/*__NETZDATEN__*/null"
KOSTEN_SCHRITT = 10   # nur jede 10. Epoche des Kostenverlaufs übernehmen (2000 statt 20000 Punkte)


def netz_daten(name, alle_eingaben):
    """
    Sammelt alles zu einem Netz.

    name:          "7seg" oder "xor" (Dateinamen in models/)
    alle_eingaben: Eingaben, für die Python Referenzvorhersagen liefert. Beim 7-Segment-Netz
                   alle 16 möglichen 4-Bit-Muster -- also auch 10-15, die nie trainiert wurden.
    """
    with open(MODELS / f"{name}.json", encoding="utf-8") as f:
        modell = json.load(f)
    protokoll = lade_protokoll(MODELS / f"{name}_training.json")
    net = NeuralNetwork.from_dict(modell)
    kosten = protokoll["kosten"]

    return {
        "modell": modell,
        "training": {
            "seed": protokoll["seed"],
            "epochs": protokoll["epochs"],
            "learning_rate": protokoll["learning_rate"],
            "dauer_s": protokoll["dauer_s"],
            "kosten_schritt": KOSTEN_SCHRITT,
            # jede 10. Epoche plus die letzte, damit die Kurve am echten Endwert endet
            "kosten": kosten[::KOSTEN_SCHRITT] + [kosten[-1]],
            "kosten_final": kosten[-1],
            "epoche_unter_0_01": next((i for i, k in enumerate(kosten) if k < 0.01), None),
        },
        "inputs": protokoll["inputs"],
        "targets": protokoll["targets"],
        "referenz_python": {
            "inputs": alle_eingaben,
            "outputs": [net.predict(x).tolist() for x in alle_eingaben],
        },
    }


def c_ausgaben():
    """
    Führt die C-Programme aus Part 4 und 5 mit --roh aus, falls sie gebaut sind.

    Rückgabe: {"txt": [[7 Werte] x 10] oder None, "header": ... oder None}
    """
    programme = {
        "txt": PROJEKT / "c_forward" / ("nn_txt.exe" if sys.platform == "win32" else "nn_txt"),
        "header": PROJEKT / "c_header_init" / ("nn_header.exe" if sys.platform == "win32" else "nn_header"),
    }
    ergebnis = {}
    for name, programm in programme.items():
        if not programm.exists():
            print(f"  C ({name}): nicht gebaut, wird übersprungen")
            ergebnis[name] = None
            continue
        lauf = subprocess.run([str(programm), "--roh"], cwd=programm.parent,
                              capture_output=True, text=True, check=False)
        # Zeilen "<ziffer> <a> ... <g>" -> nur die 7 Werte
        werte = [[float(v) for v in zeile.split()[1:]] for zeile in lauf.stdout.split("\n") if zeile.strip()]
        ergebnis[name] = werte if len(werte) == 10 else None
        print(f"  C ({name}): {len(werte)} Ziffern gelesen")
    return ergebnis


def tf_ergebnisse():
    """Ergebnisse von compare_tensorflow.py, falls vorhanden; sonst None."""
    pfad = PROJEKT / "ergebnisse" / "tf_vergleich.json"
    if not pfad.exists():
        print("  TensorFlow: ergebnisse/tf_vergleich.json fehlt, wird übersprungen")
        return None
    with open(pfad, encoding="utf-8") as f:
        return json.load(f)


def experimente():
    """
    Ergebnisse der Phase-8-Experimente (experimente/*.py), soweit vorhanden.
    Rückgabe: {"seed_studie": ..., "aktivierungen": ..., "rauschen": ..., "vektorisiert": ...};
              fehlende Experimente sind None.
    """
    ergebnis = {}
    for name in ("seed_studie", "aktivierungen", "rauschen", "vektorisiert"):
        pfad = PROJEKT / "ergebnisse" / f"{name}.json"
        if pfad.exists():
            with open(pfad, encoding="utf-8") as f:
                ergebnis[name] = json.load(f)
        else:
            print(f"  Experiment {name}: fehlt, wird übersprungen")
            ergebnis[name] = None
    return ergebnis


def ziffern_daten():
    """
    Phase 9: das Ziffernerkennungs-Netz aus train_digits.py, falls trainiert; sonst None.

    Rückgabe: {"modell", "datensatz", "test_genauigkeit", "test_anzahl", "verwechslung",
               "pixel_max" (255 bei MNIST), "test_bilder" (0..pixel_max, Ausschnitt), "test_ziffern",
               "referenz_python" (Wahrscheinlichkeiten für die ersten 20 Testbilder -- der
               Nachweis prüft, dass der Browser dasselbe rechnet)}
    """
    pfad = MODELS / "digits.json"
    if not pfad.exists():
        print("  Ziffern: models/digits.json fehlt (train_digits.py), wird übersprungen")
        return None
    with open(pfad, encoding="utf-8") as f:
        modell = json.load(f)
    protokoll = lade_protokoll(MODELS / "digits_training.json")
    net = NeuralNetwork.from_dict(modell)
    pixel_max = protokoll["pixel_max"]
    referenz = [[v / pixel_max for v in bild] for bild in protokoll["test_bilder"][:20]]
    return {
        "modell": modell,
        "datensatz": protokoll["datensatz"],
        "pixel_max": pixel_max,
        "test_anzahl": protokoll["test_anzahl"],
        "test_genauigkeit": protokoll["test_genauigkeit"],
        "train_genauigkeit": protokoll["train_genauigkeit"],
        "epochs": protokoll["epochs"],
        "dauer_s": protokoll["dauer_s"],
        "verwechslung": protokoll["verwechslung"],
        "test_bilder": protokoll["test_bilder"],
        "test_ziffern": protokoll["test_ziffern"],
        "referenz_python": {
            "inputs": referenz,
            "outputs": [net.predict(x).tolist() for x in referenz],
        },
    }


def main():
    print("Sammle Daten ...")
    daten = {
        "erzeugt": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "segmente": SEGMENT_NAMES,
        "netze": {
            # alle 16 Muster aus 4 Bit, höchstwertiges Bit zuerst: [0,0,0,0] ... [1,1,1,1]
            "7seg": netz_daten("7seg", [list(bits) for bits in product([0, 1], repeat=4)]),
            "xor": netz_daten("xor", [[0, 0], [0, 1], [1, 0], [1, 1]]),
        },
        "c": c_ausgaben(),
        "tf": tf_ergebnisse(),
        "experimente": experimente(),
        "ziffern": ziffern_daten(),
    }

    vorlage = (HIER / "vorlage.html").read_text(encoding="utf-8")
    if PLATZHALTER not in vorlage:
        raise SystemExit(f"Platzhalter {PLATZHALTER} fehlt in vorlage.html")
    # "</" maskieren: Ein "</script>" in den Daten würde sonst den <script>-Block beenden.
    json_text = json.dumps(daten, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    seite = vorlage.replace(PLATZHALTER, json_text)

    ziel = HIER / "index.html"
    ziel.write_text(seite, encoding="utf-8")
    print(f"Geschrieben: {ziel.relative_to(PROJEKT)} ({len(seite) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
