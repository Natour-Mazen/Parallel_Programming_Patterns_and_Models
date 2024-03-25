#include <4-Calcul_Pi/Pi.h>
#include <OPP/MPI/OPP_MPI.h>

#include <algorithm>
#include <cstring>
#include <iostream>

//namespace {
//    double computePartialSum(int N, int rank, int size) {
//        double sum = 0.0;
//        int start, end;
//
//        // Stratégie par bloc
//        int chunkSize = N / size;
//        start = rank * chunkSize;
//        end = (rank == size - 1) ? N : start + chunkSize;
//
//        for (int i = start; i < end; i++) {
//            double xi = (i + 0.5) / N;
//            sum += 4.0 / (1 + xi * xi);
//        }
//
//        return sum;
//    }
//}
//
//double computePi(const OPP::MPI::Ring& ring, const int N)
//{
//    int p = ring.getSize(); // nombre de processeurs
//    int rank = ring.getRank(); // rang du processeur actuel
//
//    double sum = computePartialSum(N,rank,p);
//
//    // Effectuer une opération de réduction pour additionner les sommes partielles de tous les processeurs
//    double total_sum;
//    MPI_Allreduce(&sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, ring.getComm());
//
//    // Diviser la somme totale par N pour obtenir Pi
//    double pi = total_sum / N;
//
//    return pi;
//}

namespace {
    double computePartialSum(int N, int rank, int size) {
        double sum = 0.0;
        int start, end;

        // Stratégie par bloc
        int chunkSize = N / size;
        start = rank * chunkSize;
        end = (rank == size - 1) ? N : start + chunkSize;

        for (int i = start; i < end; i++) {
            double xi = (i + 0.5) / N;
            sum += 4.0 / (1 + xi * xi);
        }

        return sum;
    }
}

double computePi(const OPP::MPI::Ring& ring, const int N)
{
    int rank = ring.getRank();
    int size = ring.getSize();

    double partialSum = computePartialSum(N, rank, size);

    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            double receivedPartialSum;
            ring.Recv(&receivedPartialSum, 1, MPI_DOUBLE);
            partialSum += receivedPartialSum;
        }
    } else {
        ring.Send(&partialSum, 1, MPI_DOUBLE);
    }

    if (rank == size - 1) {
        return partialSum / N;
    }

    return 42.0;
}