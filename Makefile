CFLAGS=-Ofast -march=native -Wall --std=gnu99 -Wno-unused-function \
	   -DSINGLE_PRECISION -DAPPROXIMATE_MATH -g -fopenmp
# This is more suitable for debugging:
#CFLAGS=-Og -Wall --std=gnu99 -Wno-unused-function -DSINGLE_PRECISION \
#		-g -fopenmp
LDFLAGS=-lm -lgomp

all: efmugal test

efmugal.o: efmugal.c natmap.c hash.c random.c
	$(CC) $(CFLAGS) -c efmugal.c

test.o: test.c natmap.c hash.c random.c
	$(CC) $(CFLAGS) -c test.c

efmugal: efmugal.o

test: test.o

clean:
	rm -f efmugal efmugal.o test test.o

