"""
Das gesamte neuronale Netz.

Bezug zur Angabe: Part 1, Step 1-3 ("NeuralNetwork class"):
    - hält eine Liste von Layer-Objekten
    - train(inputs, targets): Training mit Forward- und Backward-Pass
    - predict(inputs): Inferenz, nur Forward-Pass
Die Export-Funktionen (dump_json, load, dump_txt, dump_initfile) aus Step 4-6 liegen in
nn/persistence.py und bauen auf to_dict() / from_dict() aus dieser Datei auf. Die
gleichnamigen Methoden am Ende dieser Klasse leiten nur dorthin weiter.

Mathematik (aus der Angabe, für ein Netz mit zwei Schichten):
    Forward:   Z[1] = W[1]·X + b[1]       A[1] = f(Z[1])
               Z[2] = W[2]·A[1] + b[2]    A[2] = f(Z[2]) = Y-Dach
    Kosten:    J = 1/2 · (Y-Dach - Y)²    (Mean Squared Error, pro Sample)
    Backward:  dJ/dA[2] = A[2] - Y, dann per Kettenregel Schicht für Schicht zurück
    Update:    W_neu = W_alt - alpha · dJ/dW,   b_neu = b_alt - alpha · dJ/db

Gegenüber dem Ausgangsstand neu:
    - layer_sizes wird gespeichert (für Export und Anzeige)
    - Aktivierung pro Schicht wählbar, über den Namen (speicherbar)
    - backward() rechnet den Fehler nicht mehr durch die erste Schicht zurück
    - train() prüft die Trainingsdaten und kann die Reihenfolge der Samples mischen
    - train() kann optional Mini-Batch und Momentum (Bonus-Aufgaben); ohne diese Optionen
      läuft es über den unveränderten SGD-Weg
    - to_dict() / from_dict() als Grundlage für den JSON-Export
    - Phase 9: Kostenfunktion wählbar (loss="cross_entropy"), zusammen mit einer
      Softmax-Ausgabeschicht für die Ziffernerkennung (train_digits.py)
"""

import numpy as np

from .layer import Layer


class NeuralNetwork:
    """
    Ein Feedforward-Netz: Schichten, die hintereinander ausgeführt werden.

    Die Eingabeschicht ist KEIN eigenes Layer-Objekt -- sie ist einfach der
    Eingabevektor X (= A[0] in der Angabe). Nur Hidden- und Output-Schichten
    bestehen aus Neuronen mit Gewichten.
    """

    def __init__(self, layer_sizes, learning_rate=0.1, activation="sigmoid",
                 init="klein", layers=None, loss="mse"):
        """
        layer_sizes:   z.B. [2, 4, 1] -> 2 Eingänge, eine Hidden-Schicht mit 4 Neuronen,
                       1 Output-Neuron. Der erste Eintrag ist nur die Anzahl der
                       Eingabewerte; [2, 4, 1] ergibt also 2 Layer-Objekte.
        learning_rate: Standard-Schrittweite alpha für train()
        activation:    ein Name für alle Schichten (z.B. "sigmoid") oder eine Liste mit
                       einem Namen pro Schicht (z.B. ["tanh", "sigmoid"])
        init:          Verfahren für die Startgewichte: "klein" (Standard, 0.1 wie im
                       Ausgangsstand) oder "xavier". Warum "klein" der Standard ist,
                       steht mit Messwerten in neuron.weight_std() und CLAUDE.md (E-11).
        layers:        fertige Layer-Objekte, wenn ein Netz geladen statt neu erzeugt wird
                       (benutzt von from_dict). Dann werden activation und init ignoriert.
        loss:          Kostenfunktion für train():
                       "mse"           J = 1/2 · Summe (A - Y)²   (Angabe, Standard)
                       "cross_entropy" J = -Summe Y · log(A)      (Phase 9, für Klassen-
                                       Wahrscheinlichkeiten; Output softmax oder sigmoid)

        Fehler: ValueError bei weniger als 2 Größen, Größen < 1, falscher Anzahl an
                Aktivierungsnamen oder Layern, die nicht zu layer_sizes passen, und bei
                unbekannter Kostenfunktion bzw. cross_entropy ohne softmax/sigmoid-Ausgang.
        """
        layer_sizes = [int(size) for size in layer_sizes]
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes braucht mindestens Eingang und Ausgang, z.B. [2, 1]")
        if min(layer_sizes) < 1:
            raise ValueError(f"Jede Schicht braucht mindestens 1 Neuron: {layer_sizes}")

        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate

        # Anzahl echter Schichten = Anzahl Größen minus die Eingabe
        num_layers = len(layer_sizes) - 1

        if layers is None:
            # Ein einzelner Name gilt für alle Schichten
            if isinstance(activation, str):
                activation = [activation] * num_layers
            if len(activation) != num_layers:
                raise ValueError(
                    f"{num_layers} Schichten, aber {len(activation)} Aktivierungsnamen"
                )
            # Schicht i hat layer_sizes[i] Neuronen mit je layer_sizes[i-1] Eingängen.
            # Beispiel [2, 2, 1]:  i=1 -> Layer(2 Neuronen, je 2 Eingänge)   Hidden
            #                      i=2 -> Layer(1 Neuron,   2 Eingänge)      Output
            layers = [
                Layer(layer_sizes[i], layer_sizes[i - 1], activation[i - 1], init)
                for i in range(1, len(layer_sizes))
            ]
        else:
            # Geladene Schichten müssen zur angegebenen Struktur passen
            if len(layers) != num_layers:
                raise ValueError(f"{num_layers} Schichten erwartet, {len(layers)} bekommen")
            for i, layer in enumerate(layers, start=1):
                if (layer.num_neurons, layer.num_inputs) != (layer_sizes[i], layer_sizes[i - 1]):
                    raise ValueError(
                        f"Schicht {i}: {layer.num_neurons} Neuronen mit {layer.num_inputs} "
                        f"Eingängen passt nicht zu layer_sizes {layer_sizes}"
                    )
        self.layers = layers

        if loss not in ("mse", "cross_entropy"):
            raise ValueError(f"Unbekannte Kostenfunktion '{loss}'. Bekannt sind: mse, cross_entropy")
        if loss == "cross_entropy" and layers[-1].activation_name not in ("softmax", "sigmoid"):
            # Die Vereinfachung dJ/dZ = A - Y (siehe backward) gilt nur für diese beiden
            raise ValueError("cross_entropy braucht softmax oder sigmoid in der Output-Schicht")
        self.loss = loss

    def __repr__(self):
        """Kurzbeschreibung, z.B.  NeuralNetwork(2 -> 2 -> 1, sigmoid/sigmoid)"""
        struktur = " -> ".join(str(size) for size in self.layer_sizes)
        aktivierungen = "/".join(layer.activation_name for layer in self.layers)
        return f"NeuralNetwork({struktur}, {aktivierungen})"

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

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

        Rechnet dasselbe wie forward(). Die eigene Methode gibt es, weil die Angabe
        predict() als Schnittstelle nach außen verlangt: forward() ist der interne
        Rechenschritt beim Training, predict() ruft man mit einem fertigen Netz auf.

        inputs:   Eingabevektor X
        Rückgabe: Netzausgabe (numpy-Array, eine Zahl pro Output-Neuron)
        """
        return self.forward(inputs)

    # -----------------------------------------------------------------------
    # Backward und Training
    # -----------------------------------------------------------------------

    def backward(self, output, target, learning_rate, accumulate=False):
        """
        Ein Backward-Pass für ein einzelnes Sample.

        output:        Netzausgabe aus dem direkt vorhergehenden forward()-Aufruf
        target:        gewünschte Ausgabe Y für dieses Sample
        learning_rate: Schrittweite alpha
        accumulate:    True -> Gradienten nur sammeln, nicht anwenden (Mini-Batch, siehe
                       train und Layer.apply_gradients)

        Ablauf: Der Fehler startet an der Output-Schicht und wird Schicht für Schicht
        nach vorne zurückgereicht. Jede Schicht aktualisiert dabei ihre Gewichte und gibt
        den Fehler für ihre Vorschicht zurück.

        WICHTIG: muss direkt nach forward() für DASSELBE Sample aufgerufen werden,
        weil die Neuronen die Zwischenwerte (x, z, a) aus genau diesem Forward-Pass
        verwenden.

        Seiteneffekt: verändert alle Gewichte und Biases im Netz.
        """
        # Startfehler bei MSE: dJ/dA der Output-Schicht = A - Y  (Ableitung von 1/2 · (A - Y)²).
        # Der Faktor 1/2 in J ist genau dafür da: er kürzt sich beim Ableiten mit der 2 weg.
        #
        # Bei Cross-Entropy steht hier dieselbe Rechnung, sie bedeutet aber etwas anderes:
        # A - Y ist dann schon dJ/dZ (das delta), nicht dJ/dA. Herleitung für softmax:
        #     J = -Summe_j y_j · log(a_j)          ->  dJ/da_j = -y_j / a_j
        #     in die Softmax-Kettenregel (Layer.backward) eingesetzt:
        #     dJ/dz_i = a_i · (-y_i/a_i + Summe_j y_j) = a_i - y_i     (weil Summe y_j = 1)
        # Für sigmoid mit J = -[y·log a + (1-y)·log(1-a)] kommt ebenfalls a - y heraus.
        # Deshalb wird delta direkt übergeben -- das spart die Division durch a, die bei
        # a nahe 0 numerisch heikel wäre.
        errors = output - np.array(target, dtype=float)
        output_deltas = self.loss == "cross_entropy"

        # Von der letzten Schicht (Index len-1) rückwärts bis zur ersten (Index 0)
        last = len(self.layers) - 1
        for index in range(last, -1, -1):
            # Die erste Schicht (Index 0) hat vor sich nur die Eingabe X, die nicht lernt --
            # ihren Fehler weiterzugeben wäre verschwendete Rechenzeit.
            errors = self.layers[index].backward(
                errors, learning_rate, compute_prev_errors=(index > 0), accumulate=accumulate,
                errors_are_deltas=(output_deltas and index == last),
            )

    def cost(self, output, target):
        """
        Kosten J für ein Sample, passend zu self.loss.

        output: Netzausgabe A (Vektor), target: Sollwert Y (Vektor, gleiche Länge)
        Rückgabe: eine Zahl >= 0

        mse:           J = 1/2 · Summe (A - Y)²      (die Formel aus der Angabe)
        cross_entropy: softmax-Ausgang  J = -Summe Y · log(A)
                       sigmoid-Ausgang  J = -Summe [Y·log(A) + (1-Y)·log(1-A)]
        A wird für den Logarithmus auf [1e-15, 1 - 1e-15] begrenzt: log(0) wäre -unendlich.
        """
        if self.loss == "mse":
            return 0.5 * np.sum((output - target) ** 2)
        a = np.clip(output, 1e-15, 1 - 1e-15)
        if self.layers[-1].activation_name == "softmax":
            return -np.sum(target * np.log(a))
        return -np.sum(target * np.log(a) + (1 - target) * np.log(1 - a))

    def train(self, inputs, targets, epochs=20000, learning_rate=None,
              verbose_every=None, shuffle=False, batch_size=1, momentum=0.0):
        """
        Trainiert das Netz mit Stochastic Gradient Descent (SGD):
        pro Epoche wird jedes Sample einzeln vorwärts und rückwärts propagiert.

        inputs:        Liste von Eingabevektoren, z.B. [[0,0],[0,1],[1,0],[1,1]]
        targets:       Liste von Zielvektoren,   z.B. [[0],[1],[1],[0]]
        epochs:        wie oft der komplette Datensatz durchlaufen wird
        learning_rate: Schrittweite alpha; None -> die aus dem Konstruktor
        verbose_every: alle wie viele Epochen der Fortschritt ausgegeben wird;
                       None -> keine Ausgabe
        shuffle:       True -> Reihenfolge der Samples in jeder Epoche neu mischen
                       (benutzt den numpy-Zufallsgenerator, mit Seed reproduzierbar).
                       Standard False: feste Reihenfolge wie im Ausgangsstand und
                       wie in der Angabe beschrieben.
        batch_size:    Samples pro Gewichtsupdate (Bonus-Aufgabe "mini-batch training").
                       1 = SGD wie in der Angabe (Standard). len(inputs) = Full-Batch-
                       Gradientenabstieg. Der Gradient wird über den Batch gemittelt.
        momentum:      beta für Momentum (Bonus-Aufgabe "momentum"), z.B. 0.9.
                       0 = kein Momentum (Standard). Formel: siehe Neuron.apply_gradient.

        Rückgabe: Liste der mittleren Kosten pro Epoche -- daran sieht man, ob und wie
                  schnell das Netz lernt.

        "Stochastisch" heißt hier: Die Gewichte werden nach JEDEM einzelnen Sample
        angepasst, nicht erst nach dem ganzen Datensatz. Bei XOR sind das 4 Updates
        pro Epoche.

        WICHTIG: Mit den Standardwerten (batch_size=1, momentum=0) läuft das Training über
        den ursprünglichen SGD-Weg und ist bit-identisch mit dem Ausgangsstand. Nur wenn
        eine der beiden Optionen gesetzt ist, wird über accumulate/apply_gradients trainiert.

        Fehler: ValueError, wenn die Trainingsdaten nicht zum Netz passen oder
                batch_size < 1 bzw. momentum nicht in [0, 1) liegt.
        """
        self._check_dataset(inputs, targets)
        if batch_size < 1:
            raise ValueError(f"batch_size muss mindestens 1 sein, nicht {batch_size}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum muss in [0, 1) liegen, nicht {momentum}")
        sgd = batch_size == 1 and momentum == 0.0

        # Keine eigene Lernrate übergeben -> die aus dem Konstruktor nehmen
        if learning_rate is None:
            learning_rate = self.learning_rate

        # Reihenfolge, in der die Samples durchlaufen werden (Indizes in inputs/targets)
        order = list(range(len(inputs)))

        history = []
        for epoch in range(epochs):
            if shuffle:
                np.random.shuffle(order)

            if sgd:
                total_cost = self._sgd_epoch(inputs, targets, order, learning_rate)
            else:
                total_cost = self._batch_epoch(inputs, targets, order, learning_rate,
                                               batch_size, momentum)

            # Durchschnitt über alle Samples, damit die Zahl nicht von der
            # Größe des Datensatzes abhängt
            mean_cost = total_cost / len(inputs)
            history.append(mean_cost)

            # Fortschritt alle verbose_every Epochen ausgeben, und immer die letzte Epoche
            if verbose_every and (epoch % verbose_every == 0 or epoch == epochs - 1):
                print(f"Epoche {epoch:6d}  Kosten: {mean_cost:.6f}")

        return history

    def _sgd_epoch(self, inputs, targets, order, learning_rate):
        """
        Eine Epoche SGD: Update nach jedem einzelnen Sample (so in der Angabe).
        Rückgabe: Summe der Kosten über alle Samples.
        """
        total_cost = 0.0
        for i in order:
            # 1) Vorhersage für dieses Sample
            output = self.forward(inputs[i])
            target = np.array(targets[i], dtype=float)
            # Kosten J (Standard: 1/2 · (A - Y)², summiert über die Output-Neuronen; siehe cost)
            # (gemessen VOR dem Update, also mit dem Stand, den das Netz beim
            # Anblick dieses Samples hatte)
            total_cost += self.cost(output, target)
            # 2) Fehler zurückpropagieren und Gewichte anpassen
            self.backward(output, target, learning_rate)
        return total_cost

    def _batch_epoch(self, inputs, targets, order, learning_rate, batch_size, momentum):
        """
        Eine Epoche Mini-Batch-Gradientenabstieg, optional mit Momentum.

        Die Samples werden in Gruppen zu batch_size aufgeteilt (die letzte darf kleiner
        sein). Pro Gruppe: für jedes Sample Forward + Backward mit accumulate=True, danach
        EIN Update mit dem gemittelten Gradienten.
        Rückgabe: Summe der Kosten über alle Samples (gemessen vor dem jeweiligen Update).
        """
        total_cost = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            for i in batch:
                output = self.forward(inputs[i])
                target = np.array(targets[i], dtype=float)
                total_cost += self.cost(output, target)
                self.backward(output, target, learning_rate, accumulate=True)
            # Innerhalb des Batches ändern sich die Gewichte nicht -- alle Samples sehen
            # dasselbe Netz. Erst jetzt wird aktualisiert.
            for layer in self.layers:
                layer.apply_gradients(learning_rate, len(batch), momentum)
        return total_cost

    def _check_dataset(self, inputs, targets):
        """
        Prüft einmal vor dem Training, ob die Daten zum Netz passen.

        Ohne diese Prüfung würde ein Tippfehler in den Daten (z.B. ein Ziel mit 6 statt
        7 Segmenten) erst tief in numpy mit einer schwer verständlichen Meldung scheitern
        -- oder schlimmer: still falsch weiterrechnen.
        """
        if len(inputs) == 0:
            raise ValueError("Keine Trainingsdaten übergeben")
        if len(inputs) != len(targets):
            raise ValueError(
                f"{len(inputs)} Eingaben, aber {len(targets)} Zielwerte -- es muss zu "
                f"jeder Eingabe genau einen Zielwert geben"
            )
        for i, (x, y) in enumerate(zip(inputs, targets)):
            if len(x) != self.layer_sizes[0]:
                raise ValueError(
                    f"Sample {i}: Eingabe hat {len(x)} Werte, das Netz erwartet "
                    f"{self.layer_sizes[0]}"
                )
            if len(y) != self.layer_sizes[-1]:
                raise ValueError(
                    f"Sample {i}: Zielwert hat {len(y)} Werte, das Netz liefert "
                    f"{self.layer_sizes[-1]}"
                )

    # -----------------------------------------------------------------------
    # Export / Import (Grundlage für nn/persistence.py)
    # -----------------------------------------------------------------------

    def to_dict(self):
        """
        Das komplette Netz als verschachteltes Dictionary -- direkt JSON-tauglich.

        Aufbau:
            {"layer_sizes": [4, 16, 7],
             "learning_rate": 0.1,
             "layers": [ {"activation": "sigmoid",
                          "neurons": [ {"bias": ..., "weights": [...]}, ... ]},
                         ... ]}
        """
        data = {
            "layer_sizes": list(self.layer_sizes),
            "learning_rate": self.learning_rate,
            "layers": [layer.to_dict() for layer in self.layers],
        }
        # Die Kostenfunktion steht nur drin, wenn sie nicht der Standard ist. So bleiben die
        # Exportdateien von XOR und 7-Segment exakt wie vor Phase 9.
        if self.loss != "mse":
            data["loss"] = self.loss
        return data

    @classmethod
    def from_dict(cls, data):
        """
        Baut ein Netz aus einem Dictionary von to_dict() wieder auf.

        Verbraucht keine Zufallszahlen. Der Konstruktor prüft, dass die Schichten zu
        layer_sizes passen -- eine beschädigte Datei fällt also beim Laden auf und
        nicht erst bei der ersten Vorhersage.
        """
        layers = [Layer.from_dict(layer_data) for layer_data in data["layers"]]
        return cls(data["layer_sizes"], data.get("learning_rate", 0.1), layers=layers,
                   loss=data.get("loss", "mse"))

    # Die Angabe (Part 1) nennt dump_json/load/dump_txt/dump_initfile als Methoden der
    # NeuralNetwork-Klasse. Die eigentliche Arbeit steht in nn/persistence.py; diese
    # Methoden leiten nur weiter. Der Import steht jeweils IN der Methode, weil
    # persistence.py seinerseits NeuralNetwork braucht (sonst Zirkelimport).

    def dump_json(self, path):
        """Part 1, Step 4: speichert das Netz als JSON. Siehe persistence.dump_json."""
        from . import persistence
        return persistence.dump_json(self, path)

    @classmethod
    def load(cls, path):
        """Part 1, Step 4: lädt ein Netz aus einer JSON-Datei. Siehe persistence.load."""
        from . import persistence
        return persistence.load(path)

    def dump_txt(self, path, title="Neuronales Netz"):
        """Part 1, Step 5: menschenlesbare TXT-Datei (auch für C). Siehe persistence.dump_txt."""
        from . import persistence
        return persistence.dump_txt(self, path, title)

    @classmethod
    def load_txt(cls, path, learning_rate=0.1):
        """Lädt ein Netz aus einer TXT-Datei von dump_txt(). Siehe persistence.load_txt."""
        from . import persistence
        return persistence.load_txt(path, learning_rate)

    def dump_initfile(self, path, title="Neuronales Netz"):
        """Part 1, Step 6: C-Header zum Initialisieren. Siehe persistence.dump_initfile."""
        from . import persistence
        return persistence.dump_initfile(self, path, title)
