# $ make
# $ ./flock-sim

CC = g++
CFLAGS = -c
DEBUG = -g
SFML = -lsfml-graphics -lsfml-window -lsfml-system

all: bodies

bodies: Body.o Flock.o Pvector.o main.o Game.o
	$(CC) $(DEBUG) Body.o Flock.o Pvector.o Game.o main.o -o flock-sim $(SFML)

Body.o : Body.cpp
	$(CC) $(CFLAGS) $(DEBUG) Body.cpp

Pvector.o : Pvector.cpp
	$(CC) $(CFLAGS) $(DEBUG) Pvector.cpp

Flock.o : Flock.cpp
	$(CC) $(CFLAGS) $(DEBUG) Flock.cpp

Game.o : Game.cpp
	$(CC) $(CFLAGS) $(DEBUG) Game.cpp

clean:
	\rm *.o flock-sim

tar:
	tar cfv bodies.tar Pvector.h Pvector.cpp Body.h Body.cpp \
		Flock.h Flock.cpp Game.cpp Game.h
