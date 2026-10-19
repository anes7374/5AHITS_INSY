/*
 * nn_forward.h -- Forward-only-Netz in C: Datenstrukturen und Funktionen.
 *
 * Bezug zur Angabe:
 *   Part 4, Step 1  Structs Neuron / Layer / Network, Initialisierung aus der .txt-Datei,
 *                   forward() und predict(). KEIN Training.
 *   Part 5, Step 1  dieselben Structs und Funktionen; die Initialisierung kommt dort aus
 *                   dem generierten Header (models/7seg_net.h), der diese Datei inkludiert.
 *
 * Mathematik je Neuron (wie in Python, nn/neuron.py):
 *   z = bias + Summe_j( weights[j] * x[j] )      eine Zeile von Z[l] = W[l]·A[l-1] + b[l]
 *   output = f(z)                                 ein Eintrag von A[l] = f(Z[l])
 *
 * Speicherbesitz (gilt für alle Funktionen hier):
 *   Ein Network besitzt seine Layer, jeder Layer seine Neuronen, jedes Neuron sein
 *   weights-Array. Alles davon wird in nn_alloc() mit malloc/calloc angelegt und NUR von
 *   nn_free() freigegeben. nn_load_txt() und nn_init_from_header() rufen intern nn_alloc()
 *   auf -- ein Netz aus beiden Quellen wird also gleich freigegeben: nn_free(&net).
 */
#ifndef NN_FORWARD_H
#define NN_FORWARD_H

/*
 * Aktivierungsfunktionen. In der .txt-Datei stehen sie als Name ("sigmoid" ...), im
 * Header als diese Konstanten.
 * WICHTIG: Die Namen müssen zu nn/activations.py und C_ACTIVATION_CONSTANTS in
 * nn/persistence.py passen.
 */
typedef enum {
    NN_ACT_SIGMOID = 0,
    NN_ACT_TANH = 1,
    NN_ACT_RELU = 2
} NnActivation;

/* Ein Neuron (Angabe: "Neuron with weights, bias, output") */
typedef struct {
    double *weights;   /* Länge num_inputs; weights[j] gehört zu Eingang j. Besitz: Neuron */
    int num_inputs;    /* = Anzahl Neuronen der Vorschicht bzw. Anzahl Netzeingänge */
    double bias;
    double output;     /* Ausgabe a = f(z) aus dem letzten nn_forward()-Aufruf */
} Neuron;

/* Eine Schicht (Angabe: "Layer containing neurons") */
typedef struct {
    Neuron *neurons;          /* Länge num_neurons. Besitz: Layer */
    double *outputs;          /* Länge num_neurons: Ausgaben aller Neuronen als ein Vektor
                                 (= A[l]), Eingang der nächsten Schicht. Besitz: Layer */
    int num_neurons;
    int num_inputs;           /* gleich für alle Neuronen der Schicht */
    NnActivation activation;  /* gleich für alle Neuronen der Schicht */
} Layer;

/* Das Netz (Angabe: "Network containing layers"). Die Eingabe ist kein eigener Layer. */
typedef struct {
    Layer *layers;     /* Länge num_layers. Besitz: Network */
    int num_layers;    /* ohne Eingabeschicht, beim 7-Segment-Netz also 2 */
    int num_inputs;    /* Länge des Eingabevektors */
    int num_outputs;   /* Neuronen der letzten Schicht */
} Network;

/*
 * Legt ein Netz mit gegebener Struktur an. Alle Gewichte und Biases sind danach 0.
 *
 * net:         wird befüllt (vorheriger Inhalt wird überschrieben, NICHT freigegeben)
 * num_layers:  Anzahl Schichten ohne Eingabe (>= 1)
 * sizes:       Länge num_layers + 1; sizes[0] = Eingänge, sizes[i] = Neuronen von Schicht i-1
 * activations: Länge num_layers, Aktivierung je Schicht
 * Rückgabe:    0 bei Erfolg; -1 bei ungültigen Größen oder wenn malloc scheitert. Im
 *              Fehlerfall ist schon reservierter Speicher wieder freigegeben und net leer.
 */
int nn_alloc(Network *net, int num_layers, const int *sizes, const NnActivation *activations);

/*
 * Part 4: lädt ein Netz aus einer .txt-Datei von nn/persistence.py (dump_txt).
 *
 * Format: Zeilen, die mit '#' beginnen, und Leerzeilen werden übersprungen. Danach:
 *   <Anzahl Schichten>
 *   pro Schicht:  <Neuronen> <Eingänge> <Aktivierungsname>
 *                 pro Neuron eine Zeile: <Bias> <Gewicht 0> ... <Gewicht n-1>
 *
 * Rückgabe: 0 bei Erfolg; -1 bei Fehlern (Datei fehlt, Format falsch, kein Speicher).
 *           Die Fehlerursache wird mit Zeilennummer auf stderr ausgegeben. Im Fehlerfall
 *           ist net leer und muss nicht freigegeben werden.
 */
int nn_load_txt(Network *net, const char *path);

/*
 * Forward-Pass: rechnet die Eingabe Schicht für Schicht durch.
 *
 * input:    Länge net->num_inputs
 * Rückgabe: Zeiger auf das outputs-Array der letzten Schicht (Länge net->num_outputs).
 *           Gehört dem Netz; gültig bis zum nächsten nn_forward()/nn_free().
 *           Kann nicht scheitern: alle Puffer wurden schon in nn_alloc() angelegt.
 * Seiteneffekt: setzt neuron.output und layer.outputs in allen Schichten.
 */
const double *nn_forward(Network *net, const double *input);

/*
 * Vorhersage: Forward-Pass und Ausgabe in ein Array des Aufrufers kopieren.
 *
 * input:  Länge net->num_inputs
 * output: Länge net->num_outputs, wird vom Aufrufer bereitgestellt
 */
void nn_predict(Network *net, const double *input, double *output);

/* Gibt allen Speicher des Netzes frei und setzt es auf leer. Mehrfacher Aufruf ist sicher. */
void nn_free(Network *net);

/* Name einer Aktivierung ("sigmoid" ...), für Ausgaben */
const char *nn_activation_name(NnActivation activation);

#endif /* NN_FORWARD_H */
