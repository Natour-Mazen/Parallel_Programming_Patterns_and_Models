#include <6-Snyder/Produit.h>
#include <OPP/MPI/OPP_MPI.h>

#include <DistributedBlockMatrix.h>
#include <memory>
#include <thread>

namespace {
} // namespace

void Produit(const OPP::MPI::Torus &torus,
             const DistributedBlockMatrix &A,
             const DistributedBlockMatrix &B,
             DistributedBlockMatrix &C)
{
  // TODO
}
