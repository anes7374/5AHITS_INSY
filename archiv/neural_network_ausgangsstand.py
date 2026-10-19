"""
Neuronales Netz "from scratch" -- Ausgangsstand
===============================================

Was diese Datei macht
---------------------
Ein einfaches, vollständig verbundenes Feedforward-Netz, komplett selbst implementiert:
Forward-Pass, Backpropagation und Gradientenabstieg sind hier von Hand ausprogrammiert.
numpy wird nur für Vektorrechnung (Skalarprodukt, elementweise Operationen) und für
Zufallszahlen benutzt -- nicht für fertige NN-Bausteine.

Am Ende der Datei wird das Netz auf das XOR-Problem trainiert und das Ergebnis ausgegeben.

Bezug zur Angabe (4AHITS_NN_from_scratch.pdf)
---------------------------------------------
- Part 1, Step 1: Klassen Neuron, Layer, NeuralNetwork
- Part 1, Step 2: forward() und predict() mit Sigmoid-Aktivierung
- Part 1, Step 3: backward() und train() (Backpropagation, Stochastic Gradient Descent)
- Part 2, Step 1: XOR-Training (2-2-1, 20000 Epochen, Lernrate 0.1)
                  -> der geforderte JSON-Export fehlt in diesem Stand noch

Ausführen
---------
    python neural_network.py

Aufbau der Datei
----------------
    sigmoid, sigmoid_derivative   Aktivierungsfunktion und ihre Ableitung
    Neuron                        ein einzelnes Neuron: Gewichte, Bias, Aktivierung
    Layer                         eine Schicht aus mehreren Neuronen
    NeuralNetwork                 mehrere Schichten hintereinander + Trainingsschleife
    train_xor()                   Anwendung: XOR lernen

Schreibweise: Code <-> Angabe
-----------------------------
Die Angabe rechnet mit Matrizen für eine ganze Schicht (W[1], b[1], Z[1], A[1], ...).
Dieser Code rechnet dasselbe, aber Neuron für Neuron. Für ein einzelnes Neuron in
Schicht l gilt:

    Code                  Angabe             Bedeutung
    -------------------   ----------------   ------------------------------------------
    _last_inputs          A[l-1] bzw. X      Eingänge des Neurons (Ausgaben der Vorschicht)
    weights               eine Zeile von W[l]  ein Gewicht pro Eingang
    bias                  ein Eintrag von b[l]
    _last_raw             Z[l]               gewichtete Summe  z = w · x + b
    _last_output          A[l]               Ausgabe  a = sigma(z)
    errors (backward)     dJ/dA[l]           wie stark die Kosten J auf die Ausgabe reagieren
    delta                 dJ/dZ[l]           = dJ/dA · dA/dZ
    prev_errors           dJ/dA[l-1]         Fehler, der an die Vorschicht weitergeht

Stand: Ausgangszustand, vor dem Umbau in das Paket nn/ (siehe Implementation.md, Phase 1).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Aktivierungsfunktion
# ---------------------------------------------------------------------------

def sigmoid(x):
    """
    Sigmoid-Funktion:  sigma(x) = 1 / (1 + e^(-x))

    Quetscht jede reelle Zahl in den Bereich (0, 1):
    große positive x -> fast 1, große negative x -> fast 0, x = 0 -> genau 0.5.

    Wozu: Sie macht das Netz nichtlinear. Ohne Nichtlinearität wäre das ganze Netz,
    egal wie viele Schichten, nur eine einzige lineare Funktion -- und könnte
    XOR nicht lernen.

    x: eine Zahl oder ein numpy-Array (wird dann elementweise berechnet)
    Rückgabe: gleiche Form wie x, alle Werte zwischen 0 und 1
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Ableitung der Sigmoid-Funktion nach ihrem Eingang:

        sigma'(x) = sigma(x) * (1 - sigma(x))

    In der Angabe steht dieselbe Formel als  dA/dZ = A · (1 - A).

    x: der ROHE Eingang z (also der Wert VOR der Aktivierung), nicht die Ausgabe a.
       Deshalb wird sigmoid(x) hier noch einmal berechnet.
    Rückgabe: Steigung der Sigmoid-Kurve an der Stelle x.
              Maximal 0.25 (bei x = 0), geht gegen 0, wenn x sehr groß oder sehr klein wird.

    Hinweis: Das Neuron hat sigma(z) im Forward-Pass schon als _last_output gespeichert.
    Man könnte also direkt a * (1 - a) rechnen und sich das zweite exp() sparen.
    Das wird beim Umbau in Phase 1 geändert (Schwachstelle 3 in Ausgangsstand.md).
    """
    s = sigmoid(x)
    return s * (1.0 - s)


# ---------------------------------------------------------------------------
# Neuron
# ---------------------------------------------------------------------------

class Neuron:
    """
    Ein einzelnes künstliches Neuron.

    Rechnung im Forward-Pass:
        z = w · x + b      gewichtete Summe der Eingänge plus Bias
        a = sigma(z)       Aktivierung -> Ausgabe des Neurons

    Das Neuron merkt sich x, z und a vom letzten Aufruf, weil genau diese Werte
    beim Backward-Pass für die Ableitungen gebraucht werden.
    """

    def __init__(self, num_inputs, activation=sigmoid, activation_derivative=sigmoid_derivative):
        """
        num_inputs:            Anzahl der Eingänge = Anzahl Neuronen der Vorschicht
                               (bei der ersten Schicht: Anzahl der Eingabewerte)
        activation:            Aktivierungsfunktion, Standard: sigmoid
        activation_derivative: die passende Ableitung dazu.
                               WICHTIG: muss zur activation passen, sonst rechnet die
                               Backpropagation mit einer falschen Steigung.
        """
        self.num_inputs = num_inputs

        # Ein Gewicht pro Eingang, zufällig normalverteilt um 0 und mit 0.1 skaliert.
        # Zufällig, damit nicht alle Neuronen einer Schicht gleich starten: bei
        # identischen Startgewichten bekämen sie auch identische Gradienten und
        # würden für immer genau dasselbe lernen.
        # Die Skalierung 0.1 ergibt kleine Startwerte: langsamer, bei XOR 2-2-1 aber laut
        # Messung zuverlässiger als Xavier (7 statt 24 von 100 Läufen hängen fest).
        # (Schwachstelle 5 in Ausgangsstand.md, widerlegt -- siehe CLAUDE.md, E-11).
        self.weights = np.random.randn(num_inputs) * 0.1

        # Bias startet bei 0. Er muss nicht zufällig sein, weil die Gewichte schon
        # für unterschiedliche Startpunkte der Neuronen sorgen.
        self.bias = 0.0

        self.activation = activation
        self.activation_derivative = activation_derivative

        # Werte aus dem Forward-Pass cachen, brauchen wir für backward
        self._last_inputs = None   # x: Eingänge beim letzten activate() (Vektor, Länge num_inputs)
        self._last_raw = None      # z: gewichtete Summe vor der Aktivierung (eine Zahl)
        self._last_output = None   # a: Ausgabe nach der Aktivierung (eine Zahl zwischen 0 und 1)

    def activate(self, inputs):
        """
        Forward-Pass für dieses eine Neuron:  a = sigma(w · x + b)

        inputs:   Eingangsvektor x, Länge = num_inputs (Liste oder numpy-Array)
        Rückgabe: Ausgabe a des Neurons (eine Zahl)

        Seiteneffekt: speichert x, z und a im Neuron für den späteren Backward-Pass.
        """
        # In ein numpy-Array umwandeln, damit auch normale Python-Listen funktionieren
        # und np.dot sowie die elementweise Multiplikation in update() klappen.
        inputs = np.array(inputs)
        self._last_inputs = inputs

        # z = w · x + b   -> eine Zeile von  Z[l] = W[l] · A[l-1] + b[l]
        self._last_raw = np.dot(inputs, self.weights) + self.bias

        # a = sigma(z)    -> ein Eintrag von  A[l] = sigma(Z[l])
        self._last_output = self.activation(self._last_raw)
        return self._last_output

    def update(self, delta, learning_rate):
        """
        Gradientenabstieg für dieses Neuron:  W_neu = W_alt - alpha · dJ/dW

        delta:         dJ/dz für dieses Neuron, also wie stark die Kosten J auf die
                       gewichtete Summe z reagieren. Wird in Layer.backward berechnet.
        learning_rate: Schrittweite alpha

        Beide Gradienten ergeben sich per Kettenregel aus delta:
            dJ/dw = dJ/dz · dz/dw = delta · x     (weil z = w·x + b  ->  dz/dw = x)
            dJ/db = dJ/dz · dz/db = delta · 1     (weil dz/db = 1)
        In der Angabe entspricht das dZ[2]/dW[2] = A[1] und dZ[2]/db[2] = 1.

        WICHTIG: benutzt _last_inputs aus dem letzten activate()-Aufruf. update() darf
        also nur direkt nach dem Forward-Pass desselben Samples aufgerufen werden.

        Seiteneffekt: verändert weights und bias.
        """
        # dJ/dw = delta · x, elementweise für jedes Gewicht (x ist ein Vektor)
        self.weights -= learning_rate * delta * self._last_inputs
        # dJ/db = delta · 1
        self.bias -= learning_rate * delta


# ---------------------------------------------------------------------------
# Layer (Schicht)
# ---------------------------------------------------------------------------

class Layer:
    """
    Eine Schicht aus mehreren Neuronen, die alle dieselben Eingänge bekommen
    ("fully connected": jedes Neuron sieht jeden Ausgang der Vorschicht).

    Zusammen bilden die Gewichte aller Neuronen die Matrix W[l] aus der Angabe:
    Neuron i dieser Schicht = Zeile i von W[l].
    """

    def __init__(self, num_neurons, num_inputs_per_neuron,
                 activation=sigmoid, activation_derivative=sigmoid_derivative):
        """
        num_neurons:           wie viele Neuronen diese Schicht hat (= Anzahl ihrer Ausgänge)
        num_inputs_per_neuron: wie viele Eingänge jedes Neuron hat
                               (= Anzahl der Ausgänge der Vorschicht)
        activation, activation_derivative: werden an jedes Neuron weitergegeben
        """
        self.neurons = []
        for _ in range(num_neurons):
            neuron = Neuron(num_inputs_per_neuron, activation, activation_derivative)
            self.neurons.append(neuron)

    def forward(self, inputs):
        """
        Forward-Pass für die ganze Schicht: jedes Neuron bekommt denselben
        Eingangsvektor und liefert seine eigene Ausgabe.

        inputs:   Ausgaben der Vorschicht (bzw. die Netzeingabe X),
                  Länge = num_inputs_per_neuron
        Rückgabe: numpy-Array mit einer Ausgabe pro Neuron, Länge = num_neurons.
                  Das ist A[l] und wird zur Eingabe der nächsten Schicht.
        """
        return np.array([neuron.activate(inputs) for neuron in self.neurons])

    def backward(self, errors, learning_rate):
        """
        Backpropagation für diese Schicht.

        errors: dJ/dA für jedes Neuron dieser Schicht (Länge = Anzahl Neuronen).
                - Output-Schicht: kommt direkt aus der Kostenfunktion, A - Y
                - Hidden-Schicht: ist das prev_errors, das die NACHFOLGENDE Schicht
                  bei ihrem backward() zurückgegeben hat
        learning_rate: Schrittweite alpha für das Gewichtsupdate

        Rückgabe: dJ/dA der VORSCHICHT (Länge = Anzahl Eingänge dieser Schicht).
                  Die vorherige Schicht bekommt das als ihr 'errors' -- so wandert der
                  Fehler Schicht für Schicht nach hinten (Kettenregel).

        Seiteneffekt: aktualisiert Gewichte und Biases aller Neuronen dieser Schicht.
        """
        # 1) Deltas berechnen: delta = dJ/dA * sigma'(z)
        #    Kettenregel: dJ/dZ = dJ/dA · dA/dZ
        #    In der Angabe für die Output-Schicht:  (A[2] - Y) · A[2](1 - A[2])
        #    delta sagt, wie stark die Kosten auf die gewichtete Summe z des Neurons reagieren.
        deltas = []
        for neuron, error in zip(self.neurons, errors):
            delta = error * neuron.activation_derivative(neuron._last_raw)
            deltas.append(delta)

        # 2) Fehler für die vorherige Schicht: gewichtete Summe der Deltas.
        #    Kettenregel: prev_errors[j] = Summe über alle Neuronen i von  delta_i · w_i[j]
        #    (w_i[j] = Gewicht von Neuron i auf Eingang j; weil z_i = w_i · x + b_i,
        #    ist dz_i/dx[j] = w_i[j] -- in der Angabe dZ[2]/dA[1] = W[2]).
        #    Jeder Eingang sammelt also Fehleranteile von JEDEM Neuron dieser Schicht,
        #    gewichtet damit, wie stark das jeweilige Neuron diesen Eingang benutzt.
        #    WICHTIG: das muss mit den ALTEN Gewichten passieren, also vor dem Update,
        #    sonst stimmt die Kettenregel nicht mehr.
        # Anzahl Eingänge = Anzahl Gewichte eines Neurons (alle Neuronen haben gleich viele)
        num_inputs = len(self.neurons[0].weights)
        prev_errors = np.zeros(num_inputs)
        for neuron, delta in zip(self.neurons, deltas):
            prev_errors += delta * neuron.weights

        # 3) Erst jetzt die Gewichte und Biases aktualisieren
        #    (die Formeln dazu stehen in Neuron.update)
        for neuron, delta in zip(self.neurons, deltas):
            neuron.update(delta, learning_rate)

        return prev_errors


# ---------------------------------------------------------------------------
# NeuralNetwork (das ganze Netz)
# ---------------------------------------------------------------------------

class NeuralNetwork:
    """
    Das gesamte Netz: eine Liste von Schichten, die hintereinander ausgeführt werden.

    Die Eingabeschicht ist KEIN eigenes Layer-Objekt -- sie ist einfach der
    Eingabevektor X (= A[0] in der Angabe). Nur Hidden- und Output-Schichten
    bestehen aus Neuronen mit Gewichten.
    """

    def __init__(self, layer_sizes, learning_rate=0.1,
                 activation=sigmoid, activation_derivative=sigmoid_derivative):
        """
        layer_sizes: z.B. [2, 4, 1] -> 2 Inputs, ein Hidden-Layer mit 4 Neuronen, 1 Output
                     Der erste Eintrag ist nur die Anzahl der Eingabewerte, daraus wird
                     keine Schicht gebaut. [2, 4, 1] ergibt also 2 Layer-Objekte.
        learning_rate: Standard-Schrittweite alpha für train(), falls dort keine
                       eigene angegeben wird
        activation, activation_derivative: gelten für alle Schichten gleich
        """
        self.learning_rate = learning_rate
        self.layers = []
        # Für jede Schicht ab der ersten Hidden-Schicht: layer_sizes[i] Neuronen,
        # jedes mit so vielen Eingängen, wie die Vorschicht Ausgänge hat.
        # Beispiel [2, 2, 1]:  i=1 -> Layer(2 Neuronen, je 2 Eingänge)   Hidden
        #                      i=2 -> Layer(1 Neuron,   2 Eingänge)      Output
        for i in range(1, len(layer_sizes)):
            self.layers.append(
                Layer(layer_sizes[i], layer_sizes[i - 1], activation, activation_derivative)
            )

    def forward(self, inputs):
        """
        Forward-Pass durch das ganze Netz.

        Die Ausgabe jeder Schicht wird zur Eingabe der nächsten:
            A[0] = X  ->  A[1] = Layer1(A[0])  ->  A[2] = Layer2(A[1])  ->  ...

        inputs:   Eingabevektor X, Länge = layer_sizes[0]
        Rückgabe: Ausgabe der letzten Schicht (Y-Dach), Länge = layer_sizes[-1]

        Seiteneffekt: jedes Neuron speichert seine Zwischenwerte für backward().
        """
        # float erzwingen, damit z.B. [0, 1] (ganze Zahlen) sauber mit den
        # float-Gewichten verrechnet wird
        output = np.array(inputs, dtype=float)
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def predict(self, inputs):
        """
        Vorhersage (Inferenz): nur Forward-Pass, kein Lernen.

        Rechnet genau dasselbe wie forward(). Die eigene Methode gibt es, weil die
        Angabe predict() als Schnittstelle nach außen verlangt: forward() ist der
        interne Rechenschritt beim Training, predict() ruft man später mit einem
        fertig trainierten Netz auf.

        inputs:   Eingabevektor X
        Rückgabe: Netzausgabe (numpy-Array, eine Zahl pro Output-Neuron)
        """
        return self.forward(inputs)

    def backward(self, output, target, learning_rate):
        """
        Ein Backward-Pass für ein einzelnes Sample.
        Startfehler ist die Ableitung der Kostenfunktion J = 1/2 * (A - Y)^2,
        also dJ/dA = A - Y.

        output:        Netzausgabe aus dem direkt vorhergehenden forward()-Aufruf
        target:        gewünschte Ausgabe Y für dieses Sample
        learning_rate: Schrittweite alpha

        Ablauf: Der Fehler startet an der Output-Schicht und wird Schicht für Schicht
        nach vorne zurückgereicht (deshalb reversed). Jede Schicht aktualisiert dabei
        ihre Gewichte und gibt den Fehler für ihre Vorschicht zurück.

        WICHTIG: muss direkt nach forward() für DASSELBE Sample aufgerufen werden,
        weil die Neuronen die Zwischenwerte (x, z, a) aus genau diesem Forward-Pass
        verwenden.

        Seiteneffekt: verändert alle Gewichte und Biases im Netz.
        """
        # dJ/dA der Output-Schicht = A - Y  (Ableitung von 1/2 * (A - Y)^2).
        # Der Faktor 1/2 in J ist genau dafür da: er kürzt sich beim Ableiten mit der 2 weg.
        errors = output - np.array(target, dtype=float)
        for layer in reversed(self.layers):
            # Rückgabe = Fehler für die Vorschicht, wird im nächsten Durchlauf verwendet.
            # Bei der ersten Schicht wird das Ergebnis nicht mehr gebraucht
            # (Schwachstelle 4 in Ausgangsstand.md).
            errors = layer.backward(errors, learning_rate)

    def train(self, inputs, targets, epochs=20000, learning_rate=None, verbose_every=None):
        """
        Trainiert das Netz mit Stochastic Gradient Descent:
        pro Epoche wird jedes Sample einzeln vorwärts und rückwärts propagiert.

        inputs:  Liste von Eingabevektoren, z.B. [[0,0],[0,1],[1,0],[1,1]]
        targets: Liste von Zielvektoren,   z.B. [[0],[1],[1],[0]]
        epochs:  wie oft der komplette Datensatz durchlaufen wird
        learning_rate: Schrittweite alpha; None -> die aus dem Konstruktor
        verbose_every: alle wie viele Epochen der Fortschritt ausgegeben wird;
                       None -> keine Ausgabe
        Rückgabe: Liste der mittleren Kosten pro Epoche

        "Stochastisch" heißt hier: Die Gewichte werden nach JEDEM einzelnen Sample
        angepasst, nicht erst nach dem ganzen Datensatz. Bei XOR sind das also
        4 Updates pro Epoche. Die Reihenfolge der Samples bleibt dabei immer gleich
        (es wird nicht gemischt).
        """
        # Keine eigene Lernrate übergeben -> die aus dem Konstruktor nehmen
        if learning_rate is None:
            learning_rate = self.learning_rate

        # mittlere Kosten je Epoche -- daran sieht man, ob und wie schnell das Netz lernt
        history = []
        for epoch in range(epochs):
            total_cost = 0.0
            for x, y in zip(inputs, targets):
                # 1) Vorhersage für dieses Sample
                output = self.forward(x)
                target = np.array(y, dtype=float)
                # Kosten J = 1/2 * (A - Y)^2, summiert über die Output-Neuronen
                # (gemessen VOR dem Update, also mit dem Stand, den das Netz beim
                # Anblick dieses Samples hatte)
                total_cost += 0.5 * np.sum((output - target) ** 2)
                # 2) Fehler zurückpropagieren und Gewichte anpassen
                self.backward(output, target, learning_rate)

            # Durchschnitt über alle Samples, damit die Zahl nicht von der
            # Größe des Datensatzes abhängt
            mean_cost = total_cost / len(inputs)
            history.append(mean_cost)

            # Fortschritt alle verbose_every Epochen ausgeben, und immer die letzte Epoche
            if verbose_every and (epoch % verbose_every == 0 or epoch == epochs - 1):
                print(f"Epoche {epoch:6d}  Kosten: {mean_cost:.6f}")

        return history


# ===========================================================================
# Part 2 / Step 1: XOR-Gatter trainieren
# Architektur: 2 Input -> 2 Hidden -> 1 Output, 20000 Epochen, Lernrate 0.1
# ===========================================================================
#
# XOR liefert 1, wenn genau einer der beiden Eingänge 1 ist.
# Es ist das klassische Beispiel dafür, warum ein Netz eine Hidden-Schicht braucht:
# Die vier Punkte lassen sich nicht mit einer einzigen Geraden in "0" und "1"
# trennen. Ein Netz ohne Hidden-Schicht kann XOR deshalb grundsätzlich nicht lernen.

# Trainingsdaten XOR (Tabelle aus der Angabe, Part 2 Step 1)
X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]

# Zielwerte: jeweils eine Liste, weil das Netz einen Ausgabe-VEKTOR liefert
# (hier mit nur einem Eintrag, da es nur 1 Output-Neuron gibt)
Y = [[0],
     [1],
     [1],
     [0]]

EPOCHS = 20000         # laut Angabe
LEARNING_RATE = 0.1    # laut Angabe (alpha)


def train_xor():
    """
    Trainiert ein 2-2-1-Netz auf XOR und gibt eine Ergebnistabelle aus.

    Ablauf: Seed setzen -> Netz bauen -> trainieren -> alle 4 Eingaben testen.
    Eine Ausgabe > 0.5 zählt als 1, sonst als 0 (Schwellwert).

    Rückgabe: das trainierte Netz
    """
    # Seed setzen, damit der Lauf reproduzierbar ist.
    # Ohne Seed landet das Netz je nach Startgewichten manchmal in einem
    # lokalen Minimum und lernt XOR mit nur 2 Hidden-Neuronen nicht sauber.
    np.random.seed(1)

    # [2, 2, 1] = 2 Eingänge, 2 Hidden-Neuronen, 1 Output-Neuron
    net = NeuralNetwork([2, 2, 1], learning_rate=LEARNING_RATE)

    print(f"Training: {EPOCHS} Epochen, Lernrate {LEARNING_RATE}\n")
    # Kosten alle 2000 Epochen ausgeben -> man sieht, wie der Fehler sinkt
    net.train(X, Y, epochs=EPOCHS, verbose_every=2000)

    # Ergebnistabelle: Eingabe | Sollwert | rohe Netzausgabe | gerundet | richtig?
    print("\nErgebnisse:")
    print(f"{'Input':<10}{'Erwartet':<12}{'Ausgabe':<12}{'Gerundet':<10}{'OK'}")
    korrekt = 0
    for x, y in zip(X, Y):
        # predict liefert ein Array mit einem Eintrag -> [0] holt die Zahl heraus
        out = net.predict(x)[0]
        # Schwellwert 0.5: Sigmoid liefert nie exakt 0 oder 1, darum wird gerundet
        gerundet = int(out > 0.5)
        ok = gerundet == y[0]
        korrekt += ok   # True zählt als 1, False als 0
        print(f"{str(x):<10}{y[0]:<12}{out:<12.6f}{gerundet:<10}{'ja' if ok else 'NEIN'}")

    print(f"\n{korrekt}/{len(X)} korrekt klassifiziert")

    return net

# Nur ausführen, wenn die Datei direkt gestartet wird (python neural_network.py),
# nicht wenn sie von einer anderen Datei importiert wird.
if __name__ == "__main__":
    train_xor()
