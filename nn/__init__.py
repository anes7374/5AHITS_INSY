"""
Paket nn -- ein neuronales Netz "from scratch".

Forward-Pass, Backpropagation und Gradientenabstieg sind selbst implementiert;
numpy wird nur für Vektorrechnung und Zufallszahlen verwendet.

Aufbau:
    activations.py   Aktivierungsfunktionen + Ableitungen, Auswahl über den Namen
    neuron.py        Neuron   -- ein einzelnes Neuron
    layer.py         Layer    -- eine Schicht aus Neuronen
    network.py       NeuralNetwork -- das ganze Netz, Training und Vorhersage
    persistence.py   Speichern/Laden: JSON, TXT (auch für C), C-Header

Benutzung:
    from nn import NeuralNetwork
    net = NeuralNetwork([2, 2, 1], learning_rate=0.1)
    net.train(X, Y, epochs=20000)
    net.predict([0, 1])
    net.dump_json("models/xor.json")      # Part 1, Step 4-6: auch dump_txt, dump_initfile
    net2 = NeuralNetwork.load("models/xor.json")
"""

from . import activations, persistence
from .layer import Layer
from .network import NeuralNetwork
from .neuron import Neuron

# Was bei "from nn import *" mitkommt
__all__ = ["NeuralNetwork", "Layer", "Neuron", "activations", "persistence"]
