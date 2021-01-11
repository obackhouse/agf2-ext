CC=gcc
CDR=$(shell pwd)

CFLAGS=-Wall -Wextra -std=c11 -O3 -ffast-math
OFLAGS=-fPIC -fopenmp -lopenblas -lm
SOFLAGS=-shared -lgomp -lm -lopenblas -ggdb3
#INCLUDE=-I/home/olli/git/pyscf/pyscf/lib/vhf -I/home/olli/git/pyscf/pyscf/lib/agf2
INCLUDE=-I/home/olli/git/pyscf/pyscf/lib

all: dfragf2_slow_fast

clean:
	rm -f $(CDR)/agf2_ext/lib/*.o
	rm -f $(CDR)/agf2_ext/lib/*.so

dfragf2_slow_fast:
	$(CC) $(CFLAGS) $(INCLUDE) $(OFLAGS) -c $(CDR)/agf2_ext/lib/dfragf2_slow_fast.c -o $(CDR)/agf2_ext/lib/dfragf2_slow_fast.o
	$(CC) $(CFLAGS) $(INCLUDE) $(OFLAGS) -c $(CDR)/agf2_ext/lib/ragf2.c -o $(CDR)/agf2_ext/lib/ragf2.o
	$(CC) $(CFLAGS) $(INCLUDE) $(SOFLAGS) $(CDR)/agf2_ext/lib/ragf2.o $(CDR)/agf2_ext/lib/dfragf2_slow_fast.o -o $(CDR)/agf2_ext/lib/dfragf2_slow_fast.so
