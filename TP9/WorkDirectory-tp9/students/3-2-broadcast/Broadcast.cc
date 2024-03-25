#include <iostream>
#include <cstring>
#include <algorithm>
#include <3-2-broadcast/Broadcast.h>
#include <OPP/MPI/OPP_MPI.h>

// Version pipeline ...
void Broadcast(
    const int k, // numéro du processeur émetteur, dans 0..P-1
    int *const addr, // pointeur sur les données à envoyer/recevoir
    const int N, // nombre d'entiers à envoyer/recevoir
    const int M // taille d'un paquet de données ...
) {
    OPP::MPI::Ring ring(MPI_COMM_WORLD);
    int rank = ring.getRank();
    int p = ring.getSize();

    if(rank == k){
        for(int i=0; i<M; ++i){
            ring.Send(&addr[i*N/M], N/M, MPI_INT);
        }
    }
    else if(((rank+1)%p) == k){
        for(int i = 0; i<M; ++i){
            ring.Recv(&addr[i*N/M], N/M, MPI_INT);
        }
    }
    else {
        ring.Recv(addr, N/M, MPI_INT);
        for(int i=0; i<M-1; ++i){
            MPI_Request request = ring.AsyncSend(&addr[i*N/M], N/M, MPI_INT);
            ring.Recv(&addr[(i+1)*N/M], N/M, MPI_INT);
            MPI_Wait(&request, MPI_STATUS_IGNORE);
        }
        ring.Send(&addr[(M-1)*N/M], N/M, MPI_INT);
    }
}
