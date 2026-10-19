/*
 * test_7seg.h -- Testlauf für den 7-Segment-Decoder, gemeinsam für Part 4 und Part 5.
 *
 * Beide Parts verlangen denselben Test ("Test the network on inputs [0,0,0,0] to [1,0,0,1]
 * ... Compare the predicted outputs with expected 7-segment values"). Er steht deshalb
 * einmal hier und wird von main_txt.c (Part 4) und main_header.c (Part 5) aufgerufen.
 */
#ifndef TEST_7SEG_H
#define TEST_7SEG_H

#include "nn_forward.h"

/*
 * Testet die Ziffern 0-9 und gibt eine Tabelle aus.
 *
 * net:  ein geladenes Netz mit 4 Eingängen und 7 Ausgängen
 * raw:  0 -> lesbare Tabelle mit ASCII-Anzeige
 *       1 -> nur Rohwerte, eine Zeile pro Ziffer: "<ziffer> <a> <b> ... <g>" mit 17
 *            signifikanten Stellen. Diese Form liest tests/test_c.py, um C und Python
 *            Zahl für Zahl zu vergleichen.
 * Rückgabe: Anzahl falscher Segmente (0 = alles richtig), -1 wenn das Netz nicht 4 -> 7 ist
 */
int nn_test_7seg(Network *net, int raw);

#endif /* TEST_7SEG_H */
