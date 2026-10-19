"""
Eine Schicht (Layer) aus mehreren Neuronen.

Bezug zur Angabe: Part 1, Step 1 ("Layer class"):
    - enthält eine Liste von Neuron-Objekten
    - forward(inputs): berechnet die Ausgabe jedes Neurons der Schicht
    - backward(errors, learning_rate): Backpropagation für die Schicht,
      aktualisiert Gewichte und Biases jedes Neurons
    - gibt den Fehler für die vorherige Schicht zurück (Kettenregel)

Zusammen bilden die Gewichte aller Neuronen die Matrix W[l] aus der Angabe:
Neuron i dieser Schicht = Zeile i von W[l].

Gegenüber dem Ausgangsstand neu:
    - num_neurons, num_inputs und activation_name sind echte Felder
      (Schwachstelle 2 in Ausgangsstand.md) -- nötig für den Export
    - der Layer erzeugt die Startgewichte selbst, weil nur er beide Schichtgrößen kennt
    - backward() kann die Fehlerberechnung für die Vorschicht auslassen
      (Schwachstelle 4 in Ausgangsstand.md)
    - to_dict() / from_dict() für den Export
    - Phase 9: Softmax als Aktivierung der ganzen Schicht (forward/backward)
"""

import numpy as np

from . import activations
from .neuron import Neuron, weight_std


class Layer:
    """
    Eine vollständig verbundene Schicht ("fully connected"): jedes Neuron bekommt
    alle Ausgänge der Vorschicht als Eingänge.
    """

    def __init__(self, num_neurons, num_inputs, activation="sigmoid", init="klein",
                 neurons=None):
        """
        num_neurons: wie viele Neuronen diese Schicht hat (= Anzahl ihrer Ausgänge)
        num_inputs:  wie viele Eingänge jedes Neuron hat (= Ausgänge der Vorschicht)
        activation:  Name der Aktivierungsfunktion, gilt für alle Neuronen der Schicht
        init:        Verfahren für die Startgewichte: "klein" (Standard) oder "xavier",
                     siehe neuron.weight_std()
        neurons:     fertige Neuron-Objekte, wenn eine Schicht geladen statt neu erzeugt
                     wird (benutzt von from_dict). Dann wird init ignoriert und es werden
                     keine Zufallszahlen verbraucht.

        Fehler: ValueError, wenn übergebene Neuronen nicht zu den Größen passen.
        """
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.activation_name = activation

        if neurons is None:
            # Alle Neuronen einer Schicht ziehen ihre Gewichte mit derselben Streuung.
            std = weight_std(init, num_inputs, num_neurons)
            neurons = []
            for _ in range(num_neurons):
                # Reihenfolge der Zufallszahlen: Neuron für Neuron, je num_inputs Stück --
                # genau wie im Ausgangsstand. Mit init="klein" und gleichem Seed entstehen
                # dadurch exakt dieselben Startgewichte wie dort.
                weights = np.random.randn(num_inputs) * std
                neurons.append(Neuron(num_inputs, activation, weights=weights))
        else:
            if len(neurons) != num_neurons:
                raise ValueError(f"Erwartet {num_neurons} Neuronen, bekommen {len(neurons)}")
            for neuron in neurons:
                if neuron.num_inputs != num_inputs:
                    raise ValueError(
                        f"Neuron mit {neuron.num_inputs} Eingängen passt nicht in eine "
                        f"Schicht mit {num_inputs} Eingängen"
                    )
        self.neurons = neurons

    def forward(self, inputs):
        """
        Forward-Pass für die ganze Schicht: jedes Neuron bekommt denselben
        Eingangsvektor und liefert seine eigene Ausgabe.

        inputs:   Ausgaben der Vorschicht (bzw. die Netzeingabe X), Länge = num_inputs
        Rückgabe: numpy-Array mit einer Ausgabe pro Neuron, Länge = num_neurons.
                  Das ist A[l] und wird zur Eingabe der nächsten Schicht.

        Softmax-Schicht (Phase 9): Die Neuronen liefern nur z (Aktivierung "linear"), die
        Schicht macht daraus a = softmax(z) und trägt jedem Neuron sein a nach, damit der
        Cache (_last_output) wie bei jeder anderen Schicht A[l] enthält.
        """
        outputs = np.array([neuron.activate(inputs) for neuron in self.neurons])
        layer_fn = activations.LAYER_WISE.get(self.activation_name)
        if layer_fn is not None:
            outputs = layer_fn(outputs)
            for neuron, a in zip(self.neurons, outputs):
                neuron._last_output = a
        return outputs

    def backward(self, errors, learning_rate, compute_prev_errors=True, accumulate=False,
                 errors_are_deltas=False):
        """
        Backpropagation für diese Schicht.

        errors: dJ/dA für jedes Neuron dieser Schicht (Länge = num_neurons).
                - Output-Schicht: kommt direkt aus der Kostenfunktion, A - Y
                - Hidden-Schicht: ist das, was die NACHFOLGENDE Schicht bei ihrem
                  backward() zurückgegeben hat
        learning_rate:       Schrittweite alpha für das Gewichtsupdate
        compute_prev_errors: False für die erste Schicht -- vor ihr liegt nur die
                             Netzeingabe X, die nicht lernt. Der Fehler dorthin würde
                             nur berechnet und weggeworfen.
        accumulate:          True -> Gradienten nur sammeln statt sofort anwenden
                             (Mini-Batch/Momentum, siehe Neuron.accumulate). Angewendet
                             werden sie danach mit apply_gradients().
        errors_are_deltas:   True -> 'errors' ist schon dJ/dZ (delta) und wird nicht mehr mit
                             f'(z) multipliziert. Benutzt für die Output-Schicht bei
                             Cross-Entropy, wo sich dJ/dZ zu A - Y vereinfacht
                             (Herleitung in NeuralNetwork.backward).

        Rückgabe: dJ/dA der VORSCHICHT (Länge = num_inputs), oder None wenn
                  compute_prev_errors False ist. Die Vorschicht bekommt das als ihr
                  'errors' -- so wandert der Fehler Schicht für Schicht nach hinten.

        Seiteneffekt: aktualisiert Gewichte und Biases aller Neuronen dieser Schicht
                      (bei accumulate=True: sammelt nur die Gradienten).
        """
        # 1) Deltas berechnen: delta = dJ/dA · f'(z)
        #    Kettenregel: dJ/dZ = dJ/dA · dA/dZ
        #    In der Angabe für die Output-Schicht:  (A[2] - Y) · A[2](1 - A[2])
        #    delta sagt, wie stark die Kosten auf die gewichtete Summe z des Neurons reagieren.
        if errors_are_deltas:
            deltas = list(errors)
        elif self.activation_name in activations.LAYER_WISE:
            # Softmax: Jedes a_j hängt von ALLEN z_i ab, deshalb reicht f'(z) pro Neuron nicht.
            # Ableitung der Softmax:  da_j/dz_i = a_j · (1[i=j] - a_i)
            # Kettenregel über alle Ausgänge j:
            #     dJ/dz_i = Summe_j dJ/da_j · a_j · (1[i=j] - a_i)
            #             = a_i · (dJ/da_i - Summe_j dJ/da_j · a_j)
            a = np.array([neuron._last_output for neuron in self.neurons])
            errors = np.asarray(errors, dtype=float)
            deltas = list(a * (errors - np.dot(errors, a)))
        else:
            deltas = []
            for neuron, error in zip(self.neurons, errors):
                delta = error * neuron.derivative()
                deltas.append(delta)

        # 2) Fehler für die vorherige Schicht: gewichtete Summe der Deltas.
        #    Kettenregel: prev_errors[j] = Summe über alle Neuronen i von  delta_i · w_i[j]
        #    (w_i[j] = Gewicht von Neuron i auf Eingang j; weil z_i = w_i · x + b_i,
        #    ist dz_i/dx[j] = w_i[j] -- in der Angabe dZ[2]/dA[1] = W[2]).
        #    Jeder Eingang sammelt also Fehleranteile von JEDEM Neuron dieser Schicht,
        #    gewichtet damit, wie stark das jeweilige Neuron diesen Eingang benutzt.
        #    WICHTIG: das muss mit den ALTEN Gewichten passieren, also vor dem Update,
        #    sonst stimmt die Kettenregel nicht mehr.
        prev_errors = None
        if compute_prev_errors:
            prev_errors = np.zeros(self.num_inputs)
            for neuron, delta in zip(self.neurons, deltas):
                prev_errors += delta * neuron.weights

        # 3) Erst jetzt die Gewichte und Biases aktualisieren
        #    (die Formeln dazu stehen in Neuron.update) -- oder bei Mini-Batch nur sammeln
        for neuron, delta in zip(self.neurons, deltas):
            if accumulate:
                neuron.accumulate(delta)
            else:
                neuron.update(delta, learning_rate)

        return prev_errors

    def apply_gradients(self, learning_rate, batch_size, momentum=0.0):
        """
        Wendet die mit backward(..., accumulate=True) gesammelten Gradienten an.
        Parameter und Formel: siehe Neuron.apply_gradient.
        Seiteneffekt: verändert Gewichte und Biases aller Neuronen dieser Schicht.
        """
        for neuron in self.neurons:
            neuron.apply_gradient(learning_rate, batch_size, momentum)

    # -----------------------------------------------------------------------
    # Export / Import
    # -----------------------------------------------------------------------

    def to_dict(self):
        """
        Die Schicht als Dictionary -- direkt JSON-tauglich.

        Rückgabe: {"activation": "sigmoid", "neurons": [ {bias, weights}, ... ]}

        Die Größen (num_neurons, num_inputs) stehen nicht extra drin, sie ergeben sich
        aus der Anzahl der Neuronen und der Länge ihrer Gewichtslisten.
        """
        return {
            "activation": self.activation_name,
            "neurons": [neuron.to_dict() for neuron in self.neurons],
        }

    @classmethod
    def from_dict(cls, data):
        """
        Baut eine Schicht aus einem Dictionary von to_dict() wieder auf.

        Fehler: ValueError, wenn die Schicht leer ist oder die Neuronen unterschiedlich
                viele Gewichte haben (beschädigte oder von Hand falsch editierte Datei).
        """
        activation = data["activation"]
        neurons = [Neuron.from_dict(n, activation) for n in data["neurons"]]
        if not neurons:
            raise ValueError("Eine Schicht muss mindestens ein Neuron enthalten")
        # Die Eingangszahl des ersten Neurons gilt für die ganze Schicht;
        # der Konstruktor prüft, dass alle anderen dazu passen.
        return cls(len(neurons), neurons[0].num_inputs, activation, neurons=neurons)
