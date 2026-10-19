"""
Ein einzelnes künstliches Neuron.

Bezug zur Angabe: Part 1, Step 1 ("Neuron class"):
    - speichert seine Gewichte (weights) und seinen Bias
    - wendet eine Aktivierungsfunktion an (Standard: sigmoid)
    - berechnet  output = activation(dot(inputs, weights) + bias)
    - merkt sich die letzten Eingänge und die letzte Ausgabe für die Backpropagation
    - kann seine Gewichte und seinen Bias mit berechneten Gradienten aktualisieren

Schreibweise Code <-> Angabe (für ein Neuron in Schicht l):
    _last_inputs   = A[l-1] bzw. X   Eingänge (Ausgaben der Vorschicht)
    weights        = eine Zeile von W[l]
    bias           = ein Eintrag von b[l]
    _last_raw      = Z[l]            gewichtete Summe  z = w · x + b
    _last_output   = A[l]            Ausgabe  a = f(z)

Gegenüber dem Ausgangsstand neu:
    - activation_name: die Aktivierung wird über ihren Namen gewählt -> speicherbar (JSON)
    - Startgewichte über weight_std() wählbar: "klein" (Standard, wie im Ausgangsstand)
      oder "xavier"
    - Gewichte und Bias können beim Erzeugen direkt übergeben werden (zum Laden)
    - to_dict() / from_dict() für den Export
"""

import numpy as np

from . import activations


def weight_std(init, fan_in, fan_out):
    """
    Standardabweichung, mit der die zufälligen Startgewichte gezogen werden.

    init:    Name des Verfahrens
             "klein"  -> 0.1                            (Standard, wie im Ausgangsstand)
             "xavier" -> sqrt(2 / (fan_in + fan_out))   (Glorot/Xavier-Initialisierung)
    fan_in:  Anzahl Eingänge eines Neurons   (= Neuronen der Vorschicht)
    fan_out: Anzahl Neuronen in dieser Schicht

    Warum das eine Rolle spielt: Sind die Startgewichte sehr klein, liegen alle z nahe 0,
    alle Sigmoid-Ausgaben nahe 0.5, und die Fehler, die nach hinten weitergegeben werden,
    sind winzig -- das Netz kommt nur langsam in Gang. Sind sie zu groß, landen die z weit
    draußen im flachen Bereich der Sigmoid-Kurve, die Ableitung ist fast 0, und das Netz
    lernt ebenfalls kaum. Xavier wählt die Streuung passend zur Schichtgröße, sodass die
    Signale beim Durchlaufen des Netzes weder verschwinden noch explodieren.

    Beispiele für Xavier:  XOR 2-2-1:        Hidden 0.71, Output 0.82
                           7-Segment 4-16-7: Hidden 0.32, Output 0.29

    Warum trotzdem "klein" der Standard ist -- gemessen, nicht angenommen
    (Details in CLAUDE.md, E-11):
        XOR 2-2-1, 100 Seeds:  "klein" bleibt in  7 von 100 Läufen dauerhaft hängen,
                               "xavier" in 24 von 100. Xavier ist schneller, WENN es klappt.
        7-Segment 4-16-7:      beide 20 von 20 fehlerfrei, praktisch gleich schnell.
    Bei den Netzen dieser Angabe ist "klein" also mindestens gleich gut und bei XOR
    deutlich zuverlässiger.

    Fehler: ValueError bei unbekanntem Verfahren.
    """
    if init == "xavier":
        return np.sqrt(2.0 / (fan_in + fan_out))
    if init == "klein":
        return 0.1
    raise ValueError(f"Unbekannte Initialisierung '{init}'. Bekannt sind: xavier, klein")


class Neuron:
    """
    Ein Neuron: Forward-Pass  a = f(w · x + b)  und Gradientenschritt auf w und b.

    Das Neuron merkt sich x, z und a vom letzten activate()-Aufruf, weil genau diese
    Werte beim Backward-Pass für die Ableitungen gebraucht werden.
    """

    def __init__(self, num_inputs, activation="sigmoid", weights=None, bias=0.0):
        """
        num_inputs: Anzahl der Eingänge = Anzahl Neuronen der Vorschicht
                    (bei der ersten Schicht: Anzahl der Eingabewerte)
        activation: Name der Aktivierungsfunktion, siehe nn/activations.py
        weights:    Startgewichte (Länge num_inputs). None -> zufällig mit der
                    Standard-Initialisierung "klein" (siehe weight_std).
                    Innerhalb eines Layers erzeugt der Layer die Gewichte selbst,
                    weil nur er die Schichtgröße kennt, die "xavier" braucht.
        bias:       Start-Bias. 0 reicht, weil die Gewichte schon für unterschiedliche
                    Startpunkte der Neuronen sorgen.

        Fehler: ValueError, wenn num_inputs < 1, die Aktivierung unbekannt ist oder
                weights die falsche Länge hat.
        """
        if num_inputs < 1:
            raise ValueError(f"Ein Neuron braucht mindestens 1 Eingang, nicht {num_inputs}")

        self.num_inputs = num_inputs

        # Name UND Funktionen speichern: der Name ist für den Export,
        # die Funktionen für die eigentliche Rechnung.
        self.activation_name = activation
        self.activation, self.activation_derivative = activations.get(activation)

        if weights is None:
            weights = np.random.randn(num_inputs) * weight_std("klein", num_inputs, 1)
        # Kopie als float-Array: so kann der Aufrufer seine Liste danach weiterverwenden,
        # ohne dass sich die Gewichte des Neurons mitändern.
        weights = np.array(weights, dtype=float)
        if weights.shape != (num_inputs,):
            raise ValueError(
                f"Das Neuron hat {num_inputs} Eingänge, bekam aber {weights.size} Gewichte"
            )
        self.weights = weights
        self.bias = float(bias)

        # Cache aus dem Forward-Pass, wird von update() und Layer.backward() gebraucht
        self._last_inputs = None   # x: Eingänge beim letzten activate() (Vektor, Länge num_inputs)
        self._last_raw = None      # z: gewichtete Summe vor der Aktivierung (eine Zahl)
        self._last_output = None   # a: Ausgabe nach der Aktivierung (eine Zahl)

        # Nur für Mini-Batch/Momentum (accumulate / apply_gradient), beim SGD ungenutzt
        self._grad_w = None        # Summe der Gewichtsgradienten im Batch (Länge num_inputs)
        self._grad_b = 0.0         # Summe der Bias-Gradienten im Batch
        self._vel_w = None         # Momentum-"Geschwindigkeit" der Gewichte (Länge num_inputs)
        self._vel_b = 0.0          # Momentum-"Geschwindigkeit" des Bias

    def activate(self, inputs):
        """
        Forward-Pass für dieses Neuron:  a = f(w · x + b)

        inputs:   Eingangsvektor x, Länge = num_inputs (Liste oder numpy-Array)
        Rückgabe: Ausgabe a des Neurons (eine Zahl)

        Seiteneffekt: speichert x, z und a im Neuron für den späteren Backward-Pass.
        """
        # In ein numpy-Array umwandeln, damit auch normale Python-Listen funktionieren
        inputs = np.array(inputs)
        self._last_inputs = inputs

        # z = w · x + b   -> eine Zeile von  Z[l] = W[l] · A[l-1] + b[l]
        self._last_raw = np.dot(inputs, self.weights) + self.bias

        # a = f(z)        -> ein Eintrag von  A[l] = f(Z[l])
        self._last_output = self.activation(self._last_raw)
        return self._last_output

    def derivative(self):
        """
        Steigung der Aktivierung beim letzten Forward-Pass:  dA/dZ = f'(z)

        Nutzt die gespeicherten z und a -- es wird nichts neu berechnet.
        Bei Sigmoid ist das a · (1 - a), die Formel aus der Angabe.

        WICHTIG: nur gültig direkt nach activate() für das aktuelle Sample.
        """
        return self.activation_derivative(self._last_raw, self._last_output)

    def update(self, delta, learning_rate):
        """
        Gradientenabstieg für dieses Neuron:  W_neu = W_alt - alpha · dJ/dW

        delta:         dJ/dz für dieses Neuron -- wie stark die Kosten J auf die
                       gewichtete Summe z reagieren. Wird in Layer.backward berechnet.
        learning_rate: Schrittweite alpha

        Beide Gradienten ergeben sich per Kettenregel aus delta:
            dJ/dw = dJ/dz · dz/dw = delta · x     (weil z = w·x + b  ->  dz/dw = x)
            dJ/db = dJ/dz · dz/db = delta · 1     (weil dz/db = 1)
        In der Angabe: dZ[2]/dW[2] = A[1] und dZ[2]/db[2] = 1.

        WICHTIG: benutzt _last_inputs aus dem letzten activate(). Darf also nur direkt
        nach dem Forward-Pass desselben Samples aufgerufen werden.

        Seiteneffekt: verändert weights und bias.
        """
        # dJ/dw = delta · x, elementweise für jedes Gewicht (x ist ein Vektor)
        self.weights -= learning_rate * delta * self._last_inputs
        # dJ/db = delta · 1
        self.bias -= learning_rate * delta

    # -----------------------------------------------------------------------
    # Mini-Batch und Momentum (Bonus-Aufgabe aus Part 1)
    # -----------------------------------------------------------------------
    # Beim normalen SGD oben wird nach JEDEM Sample sofort aktualisiert. Für Mini-Batch
    # werden die Gradienten mehrerer Samples erst gesammelt (accumulate) und dann
    # gemittelt in einem Schritt angewendet (apply_gradient). Momentum merkt sich dabei
    # die letzte Änderung ("Geschwindigkeit") und rechnet sie teilweise wieder mit ein.
    # Dieser Weg ist bewusst getrennt von update(): So bleibt das Standard-Training
    # bit-identisch mit dem Ausgangsstand (tests/test_kern.py).

    def accumulate(self, delta):
        """
        Sammelt den Gradienten eines Samples, OHNE die Gewichte zu verändern.

        delta: dJ/dz für dieses Neuron und das aktuelle Sample (wie bei update)
        Seiteneffekt: addiert delta · x zur Gradientensumme und delta zur Bias-Summe.
        WICHTIG: wie update() nur direkt nach activate() für dasselbe Sample aufrufen.
        """
        if self._grad_w is None:
            self._grad_w = np.zeros(self.num_inputs)
            self._grad_b = 0.0
        self._grad_w += delta * self._last_inputs
        self._grad_b += delta

    def apply_gradient(self, learning_rate, batch_size, momentum=0.0):
        """
        Wendet die gesammelten Gradienten an (Mini-Batch-Schritt, optional mit Momentum).

        learning_rate: Schrittweite alpha
        batch_size:    Anzahl Samples, über die gesammelt wurde -> Mittelwert des Gradienten
        momentum:      beta in  v = beta · v - alpha · g,   W = W + v
                       0 = kein Momentum (reiner Mini-Batch-Gradientenabstieg).
                       Typisch 0.9: Die Änderung behält 90 % ihrer bisherigen Richtung --
                       das glättet Zickzack-Bewegungen und beschleunigt auf flachen Plateaus.

        Seiteneffekt: verändert weights und bias, setzt die Gradientensumme auf 0.
        Ohne vorheriges accumulate() passiert nichts.
        """
        if self._grad_w is None:
            return
        grad_w = self._grad_w / batch_size
        grad_b = self._grad_b / batch_size
        if self._vel_w is None:
            self._vel_w = np.zeros(self.num_inputs)
            self._vel_b = 0.0
        self._vel_w = momentum * self._vel_w - learning_rate * grad_w
        self._vel_b = momentum * self._vel_b - learning_rate * grad_b
        self.weights += self._vel_w
        self.bias += self._vel_b
        self._grad_w = None

    # -----------------------------------------------------------------------
    # Export / Import
    # -----------------------------------------------------------------------

    def to_dict(self):
        """
        Gewichte und Bias als einfaches Dictionary -- direkt JSON-tauglich.

        Rückgabe: {"bias": float, "weights": [float, ...]}

        Die Aktivierung steht NICHT hier drin, sondern einmal pro Schicht
        (siehe Layer.to_dict), weil alle Neuronen einer Schicht dieselbe haben.
        float(...) macht aus numpy-Zahlen normale Python-Zahlen, die json schreiben kann.
        """
        return {
            "bias": float(self.bias),
            "weights": [float(w) for w in self.weights],
        }

    @classmethod
    def from_dict(cls, data, activation="sigmoid"):
        """
        Baut ein Neuron aus einem Dictionary von to_dict() wieder auf.

        data:       {"bias": ..., "weights": [...]}
        activation: Name der Aktivierung (kommt aus der umgebenden Schicht)

        Verbraucht KEINE Zufallszahlen (die Gewichte werden direkt übergeben) --
        ein geladenes Netz verändert also den Zufallszustand des Programms nicht.
        """
        weights = data["weights"]
        return cls(len(weights), activation, weights=weights, bias=data["bias"])
