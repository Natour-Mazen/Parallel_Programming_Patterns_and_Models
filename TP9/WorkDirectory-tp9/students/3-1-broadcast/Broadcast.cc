#include <cstring>
#include <iostream>
#include <3-2-broadcast/Broadcast.h>
#include <OPP/MPI/OPP_MPI.h>

void Broadcast(
    const int k, // numéro du processeur émetteur, dans 0..P-1
    int *const addr, // pointeur sur les données à envoyer/recevoir
    const int N // nombre d'entiers à envoyer/recevoir
) {  
    // TODO
    OPP::MPI::Ring ring(MPI_COMM_WORLD);

    if (ring.getRank() == k) {
        ring.Send(addr, N, MPI_INT);
    } else if (ring.getNext() == k) {
        ring.Recv(addr, N, MPI_INT);
    } else {
        ring.Recv(addr, N, MPI_INT);
        ring.Send(addr, N, MPI_INT);
    }
}
