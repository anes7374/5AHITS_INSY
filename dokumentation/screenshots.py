"""
Screenshots der Visualisierung für die Quarto-Dokumentation.

Öffnet visualisierung/index.html für jedes Kapitel in Microsoft Edge ohne Fenster (headless),
im hellen Farbschema, und speichert ein PNG nach dokumentation/bilder/seite_<kapitel>.png.
Vor dem Foto kann pro Kapitel etwas JavaScript laufen (z.B. eine Ziffer zeichnen oder das
Live-Training einige tausend Epochen rechnen lassen) -- dafür wird eine Kopie der Seite mit
diesem Skript in einen temporären Ordner geschrieben; index.html selbst bleibt unverändert.
Zum Schluss wird der leere Rand unten abgeschnitten.

Voraussetzung: visualisierung/index.html ist exportiert (py -3.14 visualisierung/export_daten.py),
               Microsoft Edge ist installiert, Pillow (kommt mit matplotlib).
Ausführen:     py -3.14 dokumentation/screenshots.py
"""

import os
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

HIER = Path(__file__).resolve().parent
SEITE = HIER.parent / "visualisierung" / "index.html"
BILDER = HIER / "bilder"
BREITE, HOEHE = 1400, 2200   # Fenstergröße; die Höhe wird danach auf den Inhalt zugeschnitten

# Synthetische Striche einer "5" (Koordinaten 0..1 im Zeichenfeld) für das Kapitel "Zeichnen"
FUENF = """[[[0.7,0.17],[0.36,0.17],[0.33,0.47]].concat(Array.from({length:25},(_,i)=>{const t=-0.75*Math.PI+1.55*Math.PI*i/24;return [0.5+0.2*Math.cos(t),0.64+0.2*Math.sin(t)];}))]"""

# Kapitel-id -> JavaScript, das vor dem Foto läuft (leer = Seite wie sie ist)
KAPITEL = {
    "decoder": "S.x = bitsOf(3); update();",
    "zeichnen": f"DRAW.strokes = {FUENF}; renderDraw();",
    "rechenweg": "S.x = bitsOf(3); S.step = 3; update();",
    "neuron": "S.x = bitsOf(3); update();",
    "gewichte": "",
    "ziffern": "S.x = bitsOf(3); update();",
    "backprop": "S.x = bitsOf(3); update();",
    "training": "",
    "live": "for (let i = 0; i < 3000; i++) liveEpoch(); liveRender();",
    "xor": "",
    "nachweis": "",
    "experimente": "",
}


def edge():
    """Pfad zu msedge.exe (Standard-Installationsorte unter Windows)."""
    for basis in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles")):
        if basis and (pfad := Path(basis) / "Microsoft" / "Edge" / "Application" / "msedge.exe").exists():
            return pfad
    raise SystemExit("Microsoft Edge nicht gefunden")


def zuschneiden(pfad, rand=24):
    """Schneidet unten alle Zeilen ab, die nur Hintergrundfarbe enthalten (plus etwas Rand)."""
    bild = Image.open(pfad).convert("RGB")
    hintergrund = bild.getpixel((2, bild.height - 2))
    breite, hoehe = bild.size
    pixel = bild.load()
    letzte = 0
    for y in range(hoehe - 1, -1, -1):
        if any(pixel[x, y] != hintergrund for x in range(0, breite, 3)):
            letzte = y
            break
    bild.crop((0, 0, breite, min(hoehe, letzte + rand))).save(pfad, optimize=True)


def main():
    if not SEITE.exists():
        raise SystemExit(f"{SEITE} fehlt -- zuerst export_daten.py ausführen")
    BILDER.mkdir(exist_ok=True)
    html = SEITE.read_text(encoding="utf-8")
    browser = edge()
    with tempfile.TemporaryDirectory() as tmp:
        for kapitel, skript in KAPITEL.items():
            # Footer ausblenden; das Skript läuft nach dem Seitenaufbau (load), dann ein Bild später
            zusatz = ("<style>footer{display:none}</style><script>addEventListener('load', () => "
                      f"setTimeout(() => {{ {skript} }}, 50));</script></body>")
            kopie = Path(tmp) / f"{kapitel}.html"
            kopie.write_text(html.replace("</body>", zusatz), encoding="utf-8")
            ziel = BILDER / f"seite_{kapitel}.png"
            subprocess.run([
                str(browser), "--headless=new", "--disable-gpu", "--hide-scrollbars",
                "--blink-settings=preferredColorScheme=1",          # helles Schema (besser für Druck)
                f"--window-size={BREITE},{HOEHE}", "--virtual-time-budget=8000",
                f"--screenshot={ziel}", kopie.as_uri() + f"#{kapitel}",
            ], check=True, capture_output=True)
            zuschneiden(ziel)
            print(f"  {ziel.relative_to(HIER.parent)}")


if __name__ == "__main__":
    main()
