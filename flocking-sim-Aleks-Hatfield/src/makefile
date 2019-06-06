# $ make
# $ ./flock-sim

CC = g++
CFLAGS = -c
DEBUG = -g
SFML = -lsfml-graphics -lsfml-window -lsfml-system

all: bodies

bodies: Body.o Flock.o Pvector.o main.o Sim.o
	$(CC) $(DEBUG) Body.o Flock.o Pvector.o Sim.o main.o -o flock-sim $(SFML)

Body.o : Body.cpp
	$(CC) $(CFLAGS) $(DEBUG) Body.cpp

Pvector.o : Pvector.cpp
	$(CC) $(CFLAGS) $(DEBUG) Pvector.cpp

Flock.o : Flock.cpp
	$(CC) $(CFLAGS) $(DEBUG) Flock.cpp

Sim.o : Sim.cpp
	$(CC) $(CFLAGS) $(DEBUG) Sim.cpp

clean:
	\rm *.o flock-sim

tar:
	tar cfv bodies.tar Pvector.h Pvector.cpp Body.h Body.cpp \
		Flock.h Flock.cpp Sim.cpp Sim.h
