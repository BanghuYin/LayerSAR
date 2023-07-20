CC=gcc
CFLAGS=-I /opt/OpenBLAS/include/ 
LDFLAGS= -L/opt/OpenBLAS/lib -lopenblas -lpthread -lm
LPFLAGS= lp_dev/liblpsolve55.a -lm -ldl
DEPS=matrix.h nnet.h split.h

all: layersar 
all: CFLAGS += -O3
all: LDFLAGS += -O3

debug: layersar
debug: CFLAGS += -DDEBUG -g
debug: LDFLAGS += -DDEBUG -g

layersar: matrix.o nnet.o layersar.o split.o
	$(CC) -o $@ $^ $(LDFLAGS) $(LPFLAGS)

c.o: 
	$(CC) $(CFLAGS) $(LPFLAGS) $<  -o $@

clean:
	rm -f *.o layersar

