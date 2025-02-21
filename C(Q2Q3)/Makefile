CC = mpicc
CFLAGS = -O2 -Wall -I/opt/openblas/include
LDFLAGS = -L/opt/openblas/lib -llapacke -lopenblas -lm

all: tsqr scaling_test

tsqr: tsqr.o tsqr_lib.o
	$(CC) $(CFLAGS) -o tsqr tsqr.o tsqr_lib.o $(LDFLAGS)

scaling_test: scaling_test.o tsqr_lib.o
	$(CC) $(CFLAGS) -o scaling_test scaling_test.o tsqr_lib.o $(LDFLAGS)

run-scaling:
	@echo "[Running scaling tests]"
	mpirun -np 4 ./scaling_test > scaling_results.csv
	@echo "Data saved to scaling_results.csv"

clean:
	rm -f tsqr scaling_test *.o scaling_results.csv

tsqr.o: tsqr.c tsqr_lib.h
	$(CC) $(CFLAGS) -c tsqr.c

scaling_test.o: scaling_test.c tsqr_lib.h
	$(CC) $(CFLAGS) -c scaling_test.c

tsqr_lib.o: tsqr_lib.c tsqr_lib.h
	$(CC) $(CFLAGS) -c tsqr_lib.c

.PHONY: all clean run-scaling
