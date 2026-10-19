/*
 * main_txt.c -- Part 4: 7-Segment-Netz aus der .txt-Datei laden und testen.
 *
 * Bezug zur Angabe: Part 4, Step 3 -- "Load the .txt of a 7-seg net, test the network on
 * inputs [0,0,0,0] to [1,0,0,1], compare the predicted outputs with expected 7-segment
 * values, confirm that import retains accuracy."
 *
 * Aufruf:
 *   nn_txt [pfad] [--roh]
 *     pfad   .txt-Datei von nn/persistence.py, Standard ../models/7seg.txt
 *     --roh  nur Rohwerte ausgeben (für den automatischen Vergleich mit Python)
 *
 * Exit-Code: 0 wenn alle 70 Segmente stimmen, 1 bei falschen Segmenten, 2 bei Ladefehlern.
 * Speicher: das Netz wird von nn_load_txt() angelegt und am Ende mit nn_free() freigegeben.
 */
#include <stdio.h>
#include <string.h>

#include "nn_forward.h"
#include "test_7seg.h"

int main(int argc, char **argv)
{
    const char *path = "../models/7seg.txt";
    int raw = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--roh") == 0) {
            raw = 1;
        } else {
            path = argv[i];
        }
    }

    Network net;
    if (nn_load_txt(&net, path) != 0) {
        return 2;
    }

    if (!raw) {
        printf("Part 4 -- Netz aus %s geladen: %d", path, net.num_inputs);
        for (int l = 0; l < net.num_layers; l++) {
            printf(" -> %d", net.layers[l].num_neurons);
        }
        printf(" (%s)\n\n", nn_activation_name(net.layers[0].activation));
    }

    int wrong = nn_test_7seg(&net, raw);
    nn_free(&net);

    if (wrong < 0) {
        return 2;
    }
    return wrong == 0 ? 0 : 1;
}
