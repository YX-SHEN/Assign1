#include "tsqr_lib.h"
#include <cblas.h>
#include <lapacke.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IDX(i, j, n) ((i) * (n) + (j)) // Macro for indexing a matrix stored in row-major order

// Function to check the reconstruction error ||QR - A||_F
double check_qr_reconstruction(double *Q, double *R, double *A, int m, int n) {
    // Allocate memory for QR product
    double *QR = (double *)malloc(m * n * sizeof(double));

    // Compute QR = Q * R
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, n,
                1.0, Q, n, R, n, 0.0, QR, n);

    // Compute Frobenius norm of (QR - A)
    double error = 0.0;
    for (int i = 0; i < m * n; i++)
        error += (QR[i] - A[i]) * (QR[i] - A[i]);

    free(QR); // Free allocated memory
    return sqrt(error); // Return the reconstruction error
}

// Function to check orthogonality error ||Q^T Q - I||_F
double check_orthogonality(double *Q, int m, int n) {
    // Allocate memory for QtQ (Q^T * Q)
    double *QtQ = (double *)malloc(n * n * sizeof(double));

    // Compute QtQ = Q^T * Q
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, n, m,
                1.0, Q, n, Q, n, 0.0, QtQ, n);

    // Compute Frobenius norm of (QtQ - I)
    double error = 0.0;
    for (int i = 0; i < n; i++)
        error += (QtQ[IDX(i, i, n)] - 1.0) * (QtQ[IDX(i, i, n)] - 1.0);

    free(QtQ); // Free allocated memory
    return sqrt(error); // Return the orthogonality error
}

// Tall-Skinny QR (TSQR) implementation using MPI parallelism
void TSQR(double *A, int m, int n, double *Q_final, double *R_final, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank); // Get current process rank
    MPI_Comm_size(comm, &size); // Get total number of processes

    // Determine local matrix size for each process
    int local_m = m / size + (rank < m % size ? 1 : 0);
    
    // Allocate memory for local matrix storage
    double *local_A = (double *)malloc(local_m * n * sizeof(double));
    double *local_R = (double *)calloc(n * n, sizeof(double)); // Initialize R as zero matrix

    // Calculate scatter displacements and counts for distributing A among processes
    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    for (int i = 0, offset = 0; i < size; i++) {
        sendcounts[i] = (m / size + (i < m % size ? 1 : 0)) * n;
        displs[i] = offset;
        offset += sendcounts[i];
    }

    // Scatter matrix A across processes
    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE, 
                local_A, sendcounts[rank], MPI_DOUBLE, 0, comm);

    // Step 1: Compute local QR decomposition
    double tau[n];
    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, local_m, n, local_A, n, tau);

    // Step 2: Extract upper-triangular R factor
    for (int j = 0; j < n; j++) {
        for (int i = 0; i <= j && i < local_m; i++) {
            local_R[IDX(i, j, n)] = local_A[IDX(i, j, n)];
        }
    }
    
    // Generate Q factor
    LAPACKE_dorgqr(LAPACK_ROW_MAJOR, local_m, n, n, local_A, n, tau);

    // Step 3: Gather all R blocks at the root process
    double *R_blocks = NULL;
    if (rank == 0) {
        R_blocks = (double *)malloc(size * n * n * sizeof(double));
    }
    MPI_Gather(local_R, n * n, MPI_DOUBLE, R_blocks, n * n, MPI_DOUBLE, 0, comm);

    // Step 4: Perform global QR decomposition on gathered R blocks
    if (rank == 0) {
        double tau_global[n];
        LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, size * n, n, R_blocks, n, tau_global);
        memcpy(R_final, R_blocks, n * n * sizeof(double)); // Copy final R result

        // Generate Q matrix for merged R
        LAPACKE_dorgqr(LAPACK_ROW_MAJOR, size * n, n, n, R_blocks, n, tau_global);
    }

    // Step 5: Broadcast final R matrix to all processes
    MPI_Bcast(R_final, n * n, MPI_DOUBLE, 0, comm);

    // Step 6: Scatter merged Q blocks to all processes
    double *Q_merged_block = (double *)malloc(n * n * sizeof(double));
    MPI_Scatter(R_blocks, n * n, MPI_DOUBLE, Q_merged_block, n * n, MPI_DOUBLE, 0, comm);

    // Step 7: Compute final Q matrix using gathered Q blocks
    double *local_Q = (double *)malloc(local_m * n * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                local_m, n, n, 
                1.0, local_A, n, 
                Q_merged_block, n, 
                0.0, local_Q, n);

    // Step 8: Gather final Q matrix at root process
    MPI_Gatherv(local_Q, sendcounts[rank], MPI_DOUBLE,
               Q_final, sendcounts, displs, MPI_DOUBLE, 0, comm);

    // Step 9: Free allocated memory
    free(local_A);
    free(local_R);
    free(Q_merged_block);
    free(local_Q);
    free(sendcounts);
    free(displs);
    if (rank == 0) free(R_blocks);
}
