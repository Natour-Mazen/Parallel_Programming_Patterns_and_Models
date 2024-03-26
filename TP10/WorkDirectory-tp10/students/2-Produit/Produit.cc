#include <2-Produit/Produit.h>
#include <OPP/MPI/OPP_MPI.h>

#include <DistributedBlockVector.h>
#include <DistributedRowMatrix.h>
#include <cstring>
#include <vector>

//
/* Effectuer le calcul X = A fois B ... */
void produit(const OPP::MPI::Communicator &communicator,
             const DistributedRowMatrix &A,
             const DistributedBlockVector &B,
             DistributedBlockVector &X,
             const std::size_t N)
{
  // TODO
}
