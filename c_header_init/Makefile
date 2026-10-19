# Makefile -- Part 5: Forward-only-Netz in C, Initialisierung aus dem Header
#
# Benutzung (im Ordner c_header_init):
#   make                 baut nn_header(.exe)
#   make run             baut und testet
#   make clean           löscht das Programm
#
# Wiederverwendung statt Kopie (Part 5, Step 1): nn_forward.c und test_7seg.c kommen aus
# ../c_forward und werden einfach mitkompiliert. Der Header 7seg_net.h liegt in ../models,
# wo train_7seg.py ihn erzeugt. Wird das Netz neu trainiert, baut make automatisch neu,
# weil 7seg_net.h als Abhängigkeit eingetragen ist.

CC     = gcc
CFLAGS = -std=c11 -Wall -Wextra -O2 -I../c_forward -I../models
LDLIBS = -lm

ifeq ($(OS),Windows_NT)
  EXE = .exe
  RM  = del /Q
else
  EXE =
  RM  = rm -f
endif

PROG = nn_header$(EXE)
SRC  = main_header.c ../c_forward/nn_forward.c ../c_forward/test_7seg.c
DEPS = ../c_forward/nn_forward.h ../c_forward/test_7seg.h ../models/7seg_net.h

$(PROG): $(SRC) $(DEPS)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDLIBS)

run: $(PROG)
	./$(PROG)

clean:
	$(RM) $(PROG)

.PHONY: run clean
