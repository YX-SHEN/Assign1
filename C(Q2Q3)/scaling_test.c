#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "tsqr_lib.h"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv); // Initialize MPI environment
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the current process

    // Define the list of matrix row sizes (m) and column sizes (n) for scaling tests
    const int m_list[] = {500, 1000, 2000, 5000, 10000, 20000}; // Different row values
    const int n_list[] = {2, 5, 10, 20, 50};                    // Different column values
    const int num_m = sizeof(m_list) / sizeof(m_list[0]); // Number of row test cases
    const int num_n = sizeof(n_list) / sizeof(n_list[0]); // Number of column test cases

    // Print the CSV header in the root process
    if (rank == 0) {
        printf("m,n,time(sec)\n"); // CSV format header
        fflush(stdout); // Ensure immediate output
    }

    // Iterate over all combinations of m and n
    for (int i = 0; i < num_m; i++) {
        for (int j = 0; j < num_n; j++) {
            int m = m_list[i], n = n_list[j]; // Current matrix dimensions
            
            // Allocate memory for A, Q, and R on all processes
            double *A = (double *)malloc(m * n * sizeof(double)); // Matrix A
            double *Q = (double *)malloc(m * n * sizeof(double)); // Orthogonal matrix Q
            double *R = (double *)malloc(n * n * sizeof(double)); // Upper triangular matrix R
            
            // Check for successful memory allocation
            if (!A || !Q || !R) {
                fprintf(stderr, "Rank %d: Memory allocation failed!\n", rank);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }

            // Initialize A with random values only in the root process
            if (rank == 0) {
                srand(time(NULL)); // Seed the random number generator
                for (int k = 0; k < m * n; k++) {
                    A[k] = (double)rand() / RAND_MAX * 2.0 - 1.0; // Generate values in range [-1, 1]
                }
            }

            // Broadcast A to all processes to ensure data consistency
            MPI_Bcast(A, m * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // Start the timer for TSQR computation
            double t_start = MPI_Wtime();
            TSQR(A, m, n, Q, R, MPI_COMM_WORLD); // Perform TSQR
            double t_end = MPI_Wtime(); // Stop the timer

            // Print results only from the root process
            if (rank == 0) {
                printf("%d,%d,%.4f\n", m, n, t_end - t_start); // Output timing results
                fflush(stdout); // Ensure immediate output
            }

            // Free allocated memory
            free(A); 
            free(Q); 
            free(R);
        }
    }

    MPI_Finalize(); // Finalize MPI environment
    return 0;
}
