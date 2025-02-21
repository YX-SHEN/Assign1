#include <mpi.h> 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> 
#include "tsqr_lib.h"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv); // Initialize MPI environment

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the current process

    int m = 200, n = 10;
    double *A = NULL, *Q = NULL, *R = NULL;

    // Allocate memory for A, Q, and R on all processes
    A = (double *)malloc(m * n * sizeof(double)); // Tall-Skinny Matrix A
    R = (double *)malloc(n * n * sizeof(double)); // Upper triangular matrix R
    Q = (double *)malloc(m * n * sizeof(double)); // Orthogonal matrix Q

    // Initialize A only on the root process (rank 0)
    if (rank == 0) {
        srand(time(NULL)); // Seed random number generator
        for (int i = 0; i < m * n; i++)
            A[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0; // Generate random values in range [-1, 1]
    }

    // Broadcast matrix A to all processes to ensure data consistency
    MPI_Bcast(A, m * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform TSQR decomposition in parallel
    TSQR(A, m, n, Q, R, MPI_COMM_WORLD);

    // Compute and print numerical errors only on the root process
    if (rank == 0) {
        printf("Orthogonality error: %.6e\n", check_orthogonality(Q, m, n));
        printf("Reconstruction error: %.6e\n", check_qr_reconstruction(Q, R, A, m, n));
    }

    // Free allocated memory
    free(A);
    free(Q);
    free(R);

    MPI_Finalize(); // Finalize MPI environment
    return 0;
}
