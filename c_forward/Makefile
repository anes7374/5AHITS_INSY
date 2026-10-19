# Makefile -- Part 4: Forward-only-Netz in C, Initialisierung aus der .txt-Datei
#
# Benutzung (im Ordner c_forward):
#   make                 baut nn_txt(.exe)
#   make run             baut und testet mit ../models/7seg.txt
#   make clean           löscht das Programm
#
# Unter Windows mit MinGW-w64 heißt make meist "mingw32-make".

CC     = gcc
# -std=c11: C-Standard festlegen; -Wall -Wextra: alle sinnvollen Warnungen (wie im Plan)
CFLAGS = -std=c11 -Wall -Wextra -O2
# -lm: Mathe-Bibliothek für exp() und tanh()
LDLIBS = -lm

ifeq ($(OS),Windows_NT)
  EXE = .exe
  RM  = del /Q
else
  EXE =
  RM  = rm -f
endif

PROG = nn_txt$(EXE)
SRC  = main_txt.c nn_forward.c test_7seg.c

$(PROG): $(SRC) nn_forward.h test_7seg.h
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDLIBS)

run: $(PROG)
	./$(PROG) ../models/7seg.txt

clean:
	$(RM) $(PROG)

.PHONY: run clean
