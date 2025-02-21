// tsqr_lib.h
#ifndef TSQR_LIB_H
#define TSQR_LIB_H

#include <mpi.h>

void TSQR(double *A, int m, int n, double *Q_final, double *R_final, MPI_Comm comm);
double check_qr_reconstruction(double *Q, double *R, double *A, int m, int n);
double check_orthogonality(double *Q, int m, int n);

#endif
