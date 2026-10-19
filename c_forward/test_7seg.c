/*
 * test_7seg.c -- Testlauf für den 7-Segment-Decoder (Part 4 und Part 5, Step 3).
 * Schnittstelle: siehe test_7seg.h.
 */
#include "test_7seg.h"

#include <math.h>
#include <stdio.h>

#define NUM_DIGITS 10
#define NUM_SEGMENTS 7

/* Eingaben: die Ziffer als 4 Bit, höchstwertiges Bit zuerst (wie datasets.py) */
static const double INPUTS[NUM_DIGITS][4] = {
    {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 0, 1, 1}, {0, 1, 0, 0},
    {0, 1, 0, 1}, {0, 1, 1, 0}, {0, 1, 1, 1}, {1, 0, 0, 0}, {1, 0, 0, 1},
};

/* Soll-Segmente a-g, Tabelle von Seite 10 der Angabe (identisch mit datasets.py) */
static const int EXPECTED[NUM_DIGITS][NUM_SEGMENTS] = {
    {1, 1, 1, 1, 1, 1, 0},   /* 0 */
    {0, 1, 1, 0, 0, 0, 0},   /* 1 */
    {1, 1, 0, 1, 1, 0, 1},   /* 2 */
    {1, 1, 1, 1, 0, 0, 1},   /* 3 */
    {0, 1, 1, 0, 0, 1, 1},   /* 4 */
    {1, 0, 1, 1, 0, 1, 1},   /* 5 */
    {1, 0, 1, 1, 1, 1, 1},   /* 6 */
    {1, 1, 1, 0, 0, 0, 0},   /* 7 */
    {1, 1, 1, 1, 1, 1, 1},   /* 8 */
    {1, 1, 1, 1, 0, 1, 1},   /* 9 */
};

static const char SEGMENT_NAMES[NUM_SEGMENTS] = {'a', 'b', 'c', 'd', 'e', 'f', 'g'};

/* Segmente als Buchstaben, z.B. "ab de g" (Leerzeichen = Segment aus). out: 8 Zeichen */
static void segment_string(const int *bits, char *out)
{
    for (int s = 0; s < NUM_SEGMENTS; s++) {
        out[s] = bits[s] ? SEGMENT_NAMES[s] : ' ';
    }
    out[NUM_SEGMENTS] = '\0';
}

/*
 * Druckt alle 10 Ziffern nebeneinander als ASCII-7-Segment-Anzeige (3 Zeilen):
 *    _        Zeile 0: a
 *   |_|       Zeile 1: f g b
 *   |_|       Zeile 2: e d c
 */
static void print_ascii(int bits[NUM_DIGITS][NUM_SEGMENTS])
{
    for (int row = 0; row < 3; row++) {
        for (int d = 0; d < NUM_DIGITS; d++) {
            const int *b = bits[d];   /* b[0]=a ... b[6]=g */
            if (row == 0) {
                printf(" %c ", b[0] ? '_' : ' ');
            } else if (row == 1) {
                printf("%c%c%c", b[5] ? '|' : ' ', b[6] ? '_' : ' ', b[1] ? '|' : ' ');
            } else {
                printf("%c%c%c", b[4] ? '|' : ' ', b[3] ? '_' : ' ', b[2] ? '|' : ' ');
            }
            printf("  ");
        }
        printf("\n");
    }
}

int nn_test_7seg(Network *net, int raw)
{
    if (net->num_inputs != 4 || net->num_outputs != NUM_SEGMENTS) {
        fprintf(stderr, "Das Netz muss 4 Eingaenge und 7 Ausgaenge haben (hat %d/%d)\n",
                net->num_inputs, net->num_outputs);
        return -1;
    }

    int wrong_bits = 0;
    int wrong_digits = 0;
    int bits[NUM_DIGITS][NUM_SEGMENTS];

    if (!raw) {
        printf("Ziffer Eingabe         a      b      c      d      e      f      g    "
               "Soll     Ist      OK\n");
    }
    for (int d = 0; d < NUM_DIGITS; d++) {
        double out[NUM_SEGMENTS];
        nn_predict(net, INPUTS[d], out);

        int digit_ok = 1;
        for (int s = 0; s < NUM_SEGMENTS; s++) {
            /* Schwellwert 0.5 wie in Python (auswertung.py) */
            bits[d][s] = out[s] > 0.5;
            if (bits[d][s] != EXPECTED[d][s]) {
                wrong_bits++;
                digit_ok = 0;
            }
        }
        wrong_digits += !digit_ok;

        if (raw) {
            /* %.17g: genug Stellen, um jeden double eindeutig darzustellen */
            printf("%d", d);
            for (int s = 0; s < NUM_SEGMENTS; s++) {
                printf(" %.17g", out[s]);
            }
            printf("\n");
        } else {
            char soll[NUM_SEGMENTS + 1], ist[NUM_SEGMENTS + 1];
            segment_string(EXPECTED[d], soll);
            segment_string(bits[d], ist);
            printf("%-6d [%g, %g, %g, %g]  ", d,
                   INPUTS[d][0], INPUTS[d][1], INPUTS[d][2], INPUTS[d][3]);
            for (int s = 0; s < NUM_SEGMENTS; s++) {
                printf("%7.3f", out[s]);
            }
            printf("   %s  %s  %s\n", soll, ist, digit_ok ? "ja" : "NEIN");
        }
    }

    if (!raw) {
        printf("\nAnzeige nach Schwellwert 0.5:\n\n");
        print_ascii(bits);
        printf("\nBits richtig: %d/%d   Ziffern richtig: %d/%d\n",
               NUM_DIGITS * NUM_SEGMENTS - wrong_bits, NUM_DIGITS * NUM_SEGMENTS,
               NUM_DIGITS - wrong_digits, NUM_DIGITS);
    }
    return wrong_bits;
}
