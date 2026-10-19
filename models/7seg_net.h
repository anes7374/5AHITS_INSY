/*
 * 7seg_net.h -- 7-Segment-Decoder (Part 2, Step 3)
 * Struktur: 4 -> 16 -> 7
 *
 * AUTOMATISCH ERZEUGT von nn/persistence.py (dump_initfile). Nicht von Hand
 * bearbeiten -- stattdessen das Netz neu trainieren und exportieren.
 *
 * Bezug zur Angabe: Part 1, Step 6 (Header erzeugen) und Part 5 (Initialisierung
 * aus dem Header). Benutzung:
 *
 *     #include "7seg_net.h"
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
#ifndef NN_7SEG_NET_H
#define NN_7SEG_NET_H

#include "nn_forward.h"

/* Anzahl der Schichten ohne Eingabeschicht */
#define NN_NUM_LAYERS 2

/* Größen aller Schichten inkl. Eingabe: [0] = Eingänge, [i] = Neuronen von Schicht i-1 */
static const int NN_LAYER_SIZES[NN_NUM_LAYERS + 1] = {4, 16, 7};

/* Aktivierungsfunktion pro Schicht */
static const NnActivation NN_ACTIVATIONS[NN_NUM_LAYERS] = {NN_ACT_SIGMOID, NN_ACT_SIGMOID};

/* ---- Schicht 0 (Hidden): 16 Neuronen, je 4 Eingänge, sigmoid ---- */
static const double NN_L0_BIASES[16] = {
    -2.0722882805080385,  /* Neuron 0 */
    1.2162434281404382,  /* Neuron 1 */
    1.5704100717984475,  /* Neuron 2 */
    -0.36446124941820307,  /* Neuron 3 */
    -3.679050345623809,  /* Neuron 4 */
    1.5794639307555005,  /* Neuron 5 */
    1.8866430966339882,  /* Neuron 6 */
    -1.821043767651495,  /* Neuron 7 */
    -0.9197615602396864,  /* Neuron 8 */
    0.38532492972399124,  /* Neuron 9 */
    1.8114911434572771,  /* Neuron 10 */
    -1.7306018221062434,  /* Neuron 11 */
    -0.8422435300557982,  /* Neuron 12 */
    0.35002613201099153,  /* Neuron 13 */
    -0.13591347135894669,  /* Neuron 14 */
    -0.938657728213861,  /* Neuron 15 */
};
/* Zeile k = Gewichte von Neuron k, Spalte j = Gewicht auf Eingang j */
static const double NN_L0_WEIGHTS[16][4] = {
    {4.298265261620976, 4.665046061361925, -5.007531467257861, -0.34811915887853523},  /* Neuron 0 */
    {2.3748269172380487, -4.30334587343069, 4.380404321606658, -4.454100408788374},  /* Neuron 1 */
    {4.296648114098016, -4.845282500423819, 5.272339759382778, -4.875607943173749},  /* Neuron 2 */
    {1.3835735273860972, 0.7132121440834323, 1.6138489365975175, -0.5002728801706557},  /* Neuron 3 */
    {0.8710723220188236, 1.6529620723307812, 2.0270786069472906, 4.087496342472844},  /* Neuron 4 */
    {-1.700878846691005, 4.446716744377711, -4.2318313511864405, 1.4573670575968387},  /* Neuron 5 */
    {1.984710166770784, -1.7484835212856078, -1.76884325730477, -3.3876190029227287},  /* Neuron 6 */
    {-1.7089617759622073, -4.022355875492479, 3.5994592683958255, 3.3680343742228236},  /* Neuron 7 */
    {2.3583658597421095, 2.887203413812076, -2.992985902890028, -5.05662363760349},  /* Neuron 8 */
    {0.17872857884656715, 2.5053483095258575, -4.140079196039616, 5.190296439671948},  /* Neuron 9 */
    {-0.14968963193373128, -3.0720466012201246, -3.384813381784507, -1.078798947252232},  /* Neuron 10 */
    {0.1984607002102307, -4.540188727532718, 2.640634291898159, 3.2483986990430087},  /* Neuron 11 */
    {3.9745323460656734, 6.1464225365825325, -4.3431756577194, -2.70433898936829},  /* Neuron 12 */
    {-0.046104187907318586, -2.0399320434653494, 0.11555400921466467, 4.263248077083241},  /* Neuron 13 */
    {2.259497012552481, -1.3377146737637737, 0.22775555062017422, -1.150549553282734},  /* Neuron 14 */
    {-0.29730467803743793, -0.03499314814455858, 3.578501093290551, -0.6476345202660824},  /* Neuron 15 */
};

/* ---- Schicht 1 (Ausgabe): 7 Neuronen, je 16 Eingänge, sigmoid ---- */
static const double NN_L1_BIASES[7] = {
    -0.37370128770591515,  /* Neuron 0 */
    0.1723028017940737,  /* Neuron 1 */
    1.0800247885711476,  /* Neuron 2 */
    -0.6437412683981001,  /* Neuron 3 */
    -0.07068005776379073,  /* Neuron 4 */
    0.2240487033071082,  /* Neuron 5 */
    -0.9465518116586328,  /* Neuron 6 */
};
/* Zeile k = Gewichte von Neuron k, Spalte j = Gewicht auf Eingang j */
static const double NN_L1_WEIGHTS[7][16] = {
    {0.16872168512821148, 4.736653892768841, 5.844112052059926, 1.7432868477811965, 4.462173890132464, -2.2456580046182912, -0.5609669219778943, -1.8728769368063642, -3.7084823096397175, -2.606680395662311, -2.882888968269328, -2.8683133633472537, 1.9198075294356736, 2.9518019687170973, 0.5404839027164396, 2.867879201719088},  /* Neuron 0 */
    {-0.6739868075730062, -1.3465369439876036, -2.9753652199923373, 0.9461354061049801, -2.5050048713194, -1.1517264017303672, 3.6676333671969914, 3.447275726927321, 4.405717697348642, 3.31443630878253, 2.8599836312952274, 3.819689884314917, -4.6063504974530485, -0.8838967278026207, 2.130735179386488, 0.9831034728980574},  /* Neuron 1 */
    {1.9027966380845855, -2.5607010652785127, -1.9666798398314322, 0.23332342706813516, 2.39612699766574, 4.092057034962941, -1.5087463913556558, -0.4298126610139491, 1.1926695394776328, 4.509367874342663, 0.9513615127848504, 0.12053396874444039, 2.5486153565658474, 0.9679288823199802, -0.6248664245907799, -0.8107855674451326},  /* Neuron 2 */
    {3.240080662838016, 4.980051547240244, 5.481027476045309, -1.0047177814925257, 3.890820680309559, -2.8864638438257044, -0.4515589051230424, -2.8917936734409286, -3.601878070990701, -3.0440125389137322, 0.6176068859754826, -1.7754720998738416, 2.431118140497803, 3.1165541233560945, 0.714443083562214, -2.095511284418024},  /* Neuron 3 */
    {-1.1689135075783395, 3.7304202844522134, 4.184675144646226, 0.4809458096230562, -2.624327410315957, -1.1448394446410555, 2.9744951299217592, -2.12921333269471, 0.7141620666370433, -4.14390136421151, -0.15014205475663267, -2.4436251281497086, -0.6062296950462926, -1.5509216952888105, 0.9267684337587575, 0.7393897070696462},  /* Neuron 4 */
    {4.430911214956304, -0.5025810175866583, 1.3263027186135536, 0.611675618679939, -1.3757861497140063, 1.7423030106990827, 2.1169818638426228, -3.5673572962876046, 3.1923113570677004, 0.10943819226279553, 0.055829960014408785, -2.3017260660078724, 3.712597441772662, -1.8710727513467145, 1.0982144384238877, -1.8625893542824616},  /* Neuron 5 */
    {6.810304767684081, 2.3619674305348246, 3.2860631746204803, 0.6144111844171221, 1.694406704492529, -5.00601113993212, -2.235442438202655, 0.1605624168019744, 1.25274373978444, -1.7707719145361487, -3.9267301415577935, 2.0810585247573017, 5.705095645066746, -0.8483258553411754, 0.31103156022604683, -0.16549656035438265},  /* Neuron 6 */
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

#endif /* NN_7SEG_NET_H */
