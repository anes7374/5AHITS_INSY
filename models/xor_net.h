/*
 * xor_net.h -- XOR-Netz (Part 2, Step 1)
 * Struktur: 2 -> 2 -> 1
 *
 * AUTOMATISCH ERZEUGT von nn/persistence.py (dump_initfile). Nicht von Hand
 * bearbeiten -- stattdessen das Netz neu trainieren und exportieren.
 *
 * Bezug zur Angabe: Part 1, Step 6 (Header erzeugen) und Part 5 (Initialisierung
 * aus dem Header). Benutzung:
 *
 *     #include "xor_net.h"
 *     Network net;
 *     if (nn_init_from_header(&net) != 0) { Fehler }
 *     ... nn_predict(&net, eingabe, ausgabe) ...
 *     nn_free(&net);
 *
 * Speicher: nn_init_from_header() reserviert über nn_alloc() Speicher und kopiert die
 * Konstanten hinein. Der Aufrufer gibt ihn mit nn_free() wieder frei -- genauso wie
 * bei einem Netz aus nn_load_txt(). Beide Wege liefern also gleich gebaute Netze.
 *
 * Die Zahlen sind exakt (kürzeste Dezimaldarstellung, die der C-Compiler wieder in
 * genau denselben double-Wert übersetzt wie Python).
 */
#ifndef NN_XOR_NET_H
#define NN_XOR_NET_H

#include "nn_forward.h"

/* Anzahl der Schichten ohne Eingabeschicht */
#define NN_NUM_LAYERS 2

/* Größen aller Schichten inkl. Eingabe: [0] = Eingänge, [i] = Neuronen von Schicht i-1 */
static const int NN_LAYER_SIZES[NN_NUM_LAYERS + 1] = {2, 2, 1};

/* Aktivierungsfunktion pro Schicht */
static const NnActivation NN_ACTIVATIONS[NN_NUM_LAYERS] = {NN_ACT_SIGMOID, NN_ACT_SIGMOID};

/* ---- Schicht 0 (Hidden): 2 Neuronen, je 2 Eingänge, sigmoid ---- */
static const double NN_L0_BIASES[2] = {
    6.06586432921751,  /* Neuron 0 */
    2.2666433275357676,  /* Neuron 1 */
};
/* Zeile k = Gewichte von Neuron k, Spalte j = Gewicht auf Eingang j */
static const double NN_L0_WEIGHTS[2][2] = {
    {-4.099769290988828, -4.109435034086112},  /* Neuron 0 */
    {-5.860881390278769, -5.918335628062162},  /* Neuron 1 */
};

/* ---- Schicht 1 (Ausgabe): 1 Neuronen, je 2 Eingänge, sigmoid ---- */
static const double NN_L1_BIASES[1] = {
    -3.7342474438072357,  /* Neuron 0 */
};
/* Zeile k = Gewichte von Neuron k, Spalte j = Gewicht auf Eingang j */
static const double NN_L1_WEIGHTS[1][2] = {
    {8.083574533730397, -8.353200840676713},  /* Neuron 0 */
};

/* Zeiger auf die Arrays je Schicht, für die Schleife in nn_init_from_header() */
static const double *const NN_BIASES[NN_NUM_LAYERS] = {NN_L0_BIASES, NN_L1_BIASES};
static const double *const NN_WEIGHTS[NN_NUM_LAYERS] = {&NN_L0_WEIGHTS[0][0], &NN_L1_WEIGHTS[0][0]};

/*
 * Baut das Netz aus den Konstanten oben auf.
 *
 * net:      wird befüllt (vorheriger Inhalt wird überschrieben, nicht freigegeben)
 * Rückgabe: 0 bei Erfolg, -1 wenn nn_alloc() keinen Speicher bekommt (dann ist
 *           net leer und muss nicht freigegeben werden)
 * Speicher: gehört danach dem Aufrufer, freigeben mit nn_free(net).
 */
static int nn_init_from_header(Network *net)
{
    if (nn_alloc(net, NN_NUM_LAYERS, NN_LAYER_SIZES, NN_ACTIVATIONS) != 0) {
        return -1;
    }
    for (int l = 0; l < NN_NUM_LAYERS; l++) {
        Layer *layer = &net->layers[l];
        for (int k = 0; k < layer->num_neurons; k++) {
            Neuron *neuron = &layer->neurons[k];
            neuron->bias = NN_BIASES[l][k];
            for (int j = 0; j < neuron->num_inputs; j++) {
                /* Zeile k, Spalte j im flach abgelegten [n][m]-Array */
                neuron->weights[j] = NN_WEIGHTS[l][k * neuron->num_inputs + j];
            }
        }
    }
    return 0;
}

#endif /* NN_XOR_NET_H */
