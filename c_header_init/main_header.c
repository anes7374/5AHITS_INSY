/*
 * main_header.c -- Part 5: 7-Segment-Netz aus dem generierten Header aufbauen und testen.
 *
 * Bezug zur Angabe: Part 5 --
 *   Step 1  "Reuse forward-only network structure": Structs, nn_forward() und nn_predict()
 *           kommen unverändert aus ../c_forward/nn_forward.c (wird mitkompiliert, kein
 *           Copy-Paste). Auch der Testlauf (test_7seg.c) ist derselbe wie in Part 4.
 *   Step 2  "Use the header file for initialization": models/7seg_net.h, erzeugt von
 *           nn/persistence.py (dump_initfile), enthält Größen, Biases, Gewichte und
 *           nn_init_from_header().
 *   Step 3  Test der Ziffern 0-9.
 *
 * Der einzige Unterschied zu Part 4: Die Zahlen werden beim Kompilieren eingebaut, statt zur
 * Laufzeit aus einer Datei gelesen -- das Programm braucht keine Datei mehr (so würde es
 * auch auf einem Mikrocontroller ohne Dateisystem laufen, vgl. Part 6).
 *
 * Aufruf:    nn_header [--roh]      (--roh: nur Rohwerte, für den Vergleich mit Python)
 * Exit-Code: 0 wenn alle 70 Segmente stimmen, 1 bei falschen Segmenten, 2 bei Fehlern.
 * Speicher:  nn_init_from_header() legt das Netz über nn_alloc() an; nn_free() gibt es frei.
 */
#include <stdio.h>
#include <string.h>

#include "7seg_net.h"   /* liegt in ../models, Suchpfad setzt das Makefile (-I) */
#include "test_7seg.h"

int main(int argc, char **argv)
{
    int raw = argc > 1 && strcmp(argv[1], "--roh") == 0;

    Network net;
    if (nn_init_from_header(&net) != 0) {
        fprintf(stderr, "kein Speicher fuer das Netz\n");
        return 2;
    }

    if (!raw) {
        printf("Part 5 -- Netz aus 7seg_net.h initialisiert: %d", NN_LAYER_SIZES[0]);
        for (int l = 1; l <= NN_NUM_LAYERS; l++) {
            printf(" -> %d", NN_LAYER_SIZES[l]);
        }
        printf(" (%s)\n\n", nn_activation_name(NN_ACTIVATIONS[0]));
    }

    int wrong = nn_test_7seg(&net, raw);
    nn_free(&net);

    if (wrong < 0) {
        return 2;
    }
    return wrong == 0 ? 0 : 1;
}
