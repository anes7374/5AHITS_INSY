/*
 * nn_forward.c -- Forward-only-Netz in C: Implementierung.
 *
 * Bezug zur Angabe: Part 4, Step 1 (Initialisierung aus .txt, forward, predict) und
 * Part 5, Step 1 (dieselben Funktionen, Initialisierung aus dem Header).
 * Schnittstelle und Speicherbesitz: siehe nn_forward.h.
 *
 * Kompilieren: siehe Makefile in diesem Ordner (gcc -std=c11 -Wall -Wextra).
 */
#include "nn_forward.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------- */
/* Aktivierungsfunktionen -- dieselben Formeln wie nn/activations.py          */
/* ------------------------------------------------------------------------- */

/* sigma(z) = 1 / (1 + e^(-z)), Ausgabe in (0, 1) */
static double sigmoid(double z)
{
    return 1.0 / (1.0 + exp(-z));
}

/* relu(z) = max(0, z) */
static double relu(double z)
{
    return z > 0.0 ? z : 0.0;
}

/* Wendet die Aktivierung der Schicht an. Unbekannte Werte kommen nicht vor, weil
 * nn_alloc() und der Parser nur gültige enum-Werte zulassen. */
static double activate(NnActivation activation, double z)
{
    switch (activation) {
    case NN_ACT_TANH:
        return tanh(z);
    case NN_ACT_RELU:
        return relu(z);
    case NN_ACT_SIGMOID:
    default:
        return sigmoid(z);
    }
}

const char *nn_activation_name(NnActivation activation)
{
    switch (activation) {
    case NN_ACT_SIGMOID: return "sigmoid";
    case NN_ACT_TANH:    return "tanh";
    case NN_ACT_RELU:    return "relu";
    }
    return "?";
}

/* Name -> enum. Rückgabe 0 bei Erfolg, -1 bei unbekanntem Namen. */
static int activation_from_name(const char *name, NnActivation *out)
{
    if (strcmp(name, "sigmoid") == 0) { *out = NN_ACT_SIGMOID; return 0; }
    if (strcmp(name, "tanh") == 0)    { *out = NN_ACT_TANH;    return 0; }
    if (strcmp(name, "relu") == 0)    { *out = NN_ACT_RELU;    return 0; }
    return -1;
}

/* ------------------------------------------------------------------------- */
/* Speicher                                                                  */
/* ------------------------------------------------------------------------- */

int nn_alloc(Network *net, int num_layers, const int *sizes, const NnActivation *activations)
{
    memset(net, 0, sizeof *net);
    if (num_layers < 1) {
        return -1;
    }
    for (int i = 0; i <= num_layers; i++) {
        if (sizes[i] < 1) {
            return -1;
        }
    }

    /* calloc statt malloc: alle Zeiger sind sofort NULL. Scheitert mittendrin ein
     * malloc, kann nn_free() das halb gebaute Netz deshalb gefahrlos aufräumen. */
    net->layers = calloc((size_t)num_layers, sizeof *net->layers);
    if (net->layers == NULL) {
        return -1;
    }
    net->num_layers = num_layers;
    net->num_inputs = sizes[0];
    net->num_outputs = sizes[num_layers];

    for (int l = 0; l < num_layers; l++) {
        Layer *layer = &net->layers[l];
        /* Schicht l hat sizes[l+1] Neuronen mit je sizes[l] Eingängen
         * (sizes[0] ist die Eingabe, siehe nn_forward.h) */
        layer->num_neurons = sizes[l + 1];
        layer->num_inputs = sizes[l];
        layer->activation = activations[l];
        layer->neurons = calloc((size_t)layer->num_neurons, sizeof *layer->neurons);
        layer->outputs = calloc((size_t)layer->num_neurons, sizeof *layer->outputs);
        if (layer->neurons == NULL || layer->outputs == NULL) {
            nn_free(net);
            return -1;
        }
        for (int k = 0; k < layer->num_neurons; k++) {
            Neuron *neuron = &layer->neurons[k];
            neuron->num_inputs = layer->num_inputs;
            neuron->weights = calloc((size_t)neuron->num_inputs, sizeof *neuron->weights);
            if (neuron->weights == NULL) {
                nn_free(net);
                return -1;
            }
        }
    }
    return 0;
}

void nn_free(Network *net)
{
    if (net->layers != NULL) {
        for (int l = 0; l < net->num_layers; l++) {
            Layer *layer = &net->layers[l];
            if (layer->neurons != NULL) {
                for (int k = 0; k < layer->num_neurons; k++) {
                    free(layer->neurons[k].weights);   /* free(NULL) ist erlaubt */
                }
            }
            free(layer->neurons);
            free(layer->outputs);
        }
        free(net->layers);
    }
    /* Auf leer setzen: ein zweiter nn_free()-Aufruf tut dann nichts mehr */
    memset(net, 0, sizeof *net);
}

/* ------------------------------------------------------------------------- */
/* Part 4: Laden aus der .txt-Datei                                          */
/* ------------------------------------------------------------------------- */

/* Längste erlaubte Zeile. 16 Gewichte à 24 Zeichen brauchen ~420 Zeichen; die Grenze
 * reicht für Schichten mit über 300 Eingängen. Längere Zeilen werden als Fehler gemeldet,
 * nicht stillschweigend abgeschnitten. */
#define MAX_LINE 8192

/* Zustand des Parsers: Datei und aktuelle Zeilennummer (für Fehlermeldungen) */
typedef struct {
    FILE *file;
    const char *path;
    int line_no;
    char line[MAX_LINE];
} Reader;

/*
 * Liest die nächste Datenzeile nach reader->line: überspringt Leerzeilen und Zeilen,
 * deren erstes sichtbares Zeichen '#' ist -- dieselbe Regel wie load_txt() in Python.
 * Rückgabe: 0 bei Erfolg, -1 bei Dateiende oder zu langer Zeile (mit Meldung).
 */
static int next_data_line(Reader *reader, const char *expected)
{
    while (fgets(reader->line, sizeof reader->line, reader->file) != NULL) {
        reader->line_no++;
        size_t len = strlen(reader->line);
        if (len == sizeof reader->line - 1 && reader->line[len - 1] != '\n' && !feof(reader->file)) {
            fprintf(stderr, "%s:%d: Zeile zu lang (max. %d Zeichen)\n",
                    reader->path, reader->line_no, MAX_LINE - 2);
            return -1;
        }
        const char *p = reader->line;
        while (isspace((unsigned char)*p)) {
            p++;
        }
        if (*p != '\0' && *p != '#') {
            return 0;
        }
    }
    fprintf(stderr, "%s: Datei endet zu frueh, erwartet: %s\n", reader->path, expected);
    return -1;
}

/*
 * Liest genau 'count' Zahlen aus einer Zeile.
 * strtod liest die von Python (repr) geschriebenen Zahlen exakt -- dadurch rechnet C mit
 * bit-identischen Gewichten.
 * Rückgabe: 0 bei Erfolg, -1 wenn zu wenige, zu viele oder ungültige Zahlen dastehen.
 */
static int parse_numbers(Reader *reader, double *values, int count)
{
    char *p = reader->line;
    for (int i = 0; i < count; i++) {
        char *end;
        errno = 0;
        values[i] = strtod(p, &end);
        if (end == p || errno == ERANGE) {
            fprintf(stderr, "%s:%d: Zahl %d von %d fehlt oder ist ungueltig\n",
                    reader->path, reader->line_no, i + 1, count);
            return -1;
        }
        p = end;
    }
    while (isspace((unsigned char)*p)) {
        p++;
    }
    if (*p != '\0') {
        fprintf(stderr, "%s:%d: mehr als %d Zahlen in der Zeile\n",
                reader->path, reader->line_no, count);
        return -1;
    }
    return 0;
}

int nn_load_txt(Network *net, const char *path)
{
    memset(net, 0, sizeof *net);

    Reader reader = {0};
    reader.path = path;
    reader.file = fopen(path, "r");
    if (reader.file == NULL) {
        fprintf(stderr, "%s: Datei kann nicht geoeffnet werden\n", path);
        return -1;
    }

    /* Die Datei beschreibt die Größen erst Schicht für Schicht, nn_alloc() braucht sie aber
     * alle vorher. Deshalb zwei Durchgänge: (1) nur die Kopfzeilen lesen und Gewichtszeilen
     * überspringen, (2) Speicher anlegen, Datei von vorn lesen und die Zahlen eintragen. */
    int num_layers = 0;
    int *sizes = NULL;
    NnActivation *activations = NULL;
    int result = -1;

    if (next_data_line(&reader, "Anzahl der Schichten") != 0) {
        goto done;
    }
    if (sscanf(reader.line, "%d", &num_layers) != 1 || num_layers < 1) {
        fprintf(stderr, "%s:%d: ungueltige Anzahl der Schichten\n", path, reader.line_no);
        goto done;
    }
    sizes = calloc((size_t)num_layers + 1, sizeof *sizes);
    activations = calloc((size_t)num_layers, sizeof *activations);
    if (sizes == NULL || activations == NULL) {
        fprintf(stderr, "kein Speicher\n");
        goto done;
    }

    /* Durchgang 1: Struktur */
    for (int l = 0; l < num_layers; l++) {
        int neurons, inputs;
        char name[32];
        if (next_data_line(&reader, "Kopfzeile einer Schicht") != 0) {
            goto done;
        }
        if (sscanf(reader.line, "%d %d %31s", &neurons, &inputs, name) != 3
            || neurons < 1 || inputs < 1) {
            fprintf(stderr, "%s:%d: Kopfzeile muss '<Neuronen> <Eingaenge> <Aktivierung>' sein\n",
                    path, reader.line_no);
            goto done;
        }
        if (activation_from_name(name, &activations[l]) != 0) {
            fprintf(stderr, "%s:%d: unbekannte Aktivierung '%s'\n", path, reader.line_no, name);
            goto done;
        }
        /* Die Eingänge einer Schicht müssen die Neuronen der Vorschicht sein */
        if (l > 0 && inputs != sizes[l]) {
            fprintf(stderr, "%s:%d: Schicht %d hat %d Eingaenge, Vorschicht aber %d Neuronen\n",
                    path, reader.line_no, l, inputs, sizes[l]);
            goto done;
        }
        sizes[l] = inputs;
        sizes[l + 1] = neurons;
        for (int k = 0; k < neurons; k++) {
            if (next_data_line(&reader, "Gewichtszeile eines Neurons") != 0) {
                goto done;
            }
        }
    }

    if (nn_alloc(net, num_layers, sizes, activations) != 0) {
        fprintf(stderr, "kein Speicher fuer das Netz\n");
        goto done;
    }

    /* Durchgang 2: Zahlen. Die Struktur ist schon geprüft, darum nur noch die Werte. */
    rewind(reader.file);
    reader.line_no = 0;
    next_data_line(&reader, "Anzahl der Schichten");
    for (int l = 0; l < num_layers; l++) {
        Layer *layer = &net->layers[l];
        double *values = malloc(((size_t)layer->num_inputs + 1) * sizeof *values);
        if (values == NULL) {
            fprintf(stderr, "kein Speicher\n");
            nn_free(net);
            goto done;
        }
        next_data_line(&reader, "Kopfzeile einer Schicht");
        for (int k = 0; k < layer->num_neurons; k++) {
            Neuron *neuron = &layer->neurons[k];
            next_data_line(&reader, "Gewichtszeile eines Neurons");
            /* Zeile = Bias, dann num_inputs Gewichte */
            if (parse_numbers(&reader, values, neuron->num_inputs + 1) != 0) {
                free(values);
                nn_free(net);
                goto done;
            }
            neuron->bias = values[0];
            memcpy(neuron->weights, values + 1, (size_t)neuron->num_inputs * sizeof *values);
        }
        free(values);
    }
    result = 0;

done:
    free(sizes);
    free(activations);
    fclose(reader.file);
    return result;
}

/* ------------------------------------------------------------------------- */
/* Forward und Predict                                                       */
/* ------------------------------------------------------------------------- */

const double *nn_forward(Network *net, const double *input)
{
    /* x = Eingang der aktuellen Schicht: zuerst die Netzeingabe X = A[0],
     * danach jeweils die Ausgabe A[l] der Vorschicht */
    const double *x = input;
    for (int l = 0; l < net->num_layers; l++) {
        Layer *layer = &net->layers[l];
        for (int k = 0; k < layer->num_neurons; k++) {
            Neuron *neuron = &layer->neurons[k];
            /* z = w · x + b  -- eine Zeile von Z[l] = W[l]·A[l-1] + b[l] */
            double z = 0.0;
            for (int j = 0; j < neuron->num_inputs; j++) {
                z += neuron->weights[j] * x[j];
            }
            z += neuron->bias;
            /* a = f(z) */
            neuron->output = activate(layer->activation, z);
            layer->outputs[k] = neuron->output;
        }
        /* WICHTIG: erst nach der kompletten Schicht umschalten -- alle Neuronen einer
         * Schicht müssen denselben Eingang sehen. */
        x = layer->outputs;
    }
    return x;
}

void nn_predict(Network *net, const double *input, double *output)
{
    const double *result = nn_forward(net, input);
    memcpy(output, result, (size_t)net->num_outputs * sizeof *output);
}
