#include <iostream>
#include <cstring>
#include <algorithm>
#include <3-2-broadcast/Broadcast.h>
#include <MPI/OPP_MPI.h>

// Version pipeline ...
void Broadcast(
    const int k, // numéro du processeur émetteur, dans 0..P-1
    int *const addr, // pointeur sur les données à envoyer/recevoir
    const int N, // nombre d'entiers à envoyer/recevoir
    const int M // taille d'un paquet de données ...
) {  
    // TODO
}
