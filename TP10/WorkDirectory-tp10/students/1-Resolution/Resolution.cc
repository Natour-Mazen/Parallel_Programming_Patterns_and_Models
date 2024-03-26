#include <1-Resolution/Resolution.h>
#include <OPP/MPI/OPP_MPI.h>

#include <algorithm>
namespace {
  // Cette fonction résout un bloc du système
  void solveBlock(const DistributedRowMatrix &L,
                  DistributedBlockVector &X,
                  float* newB,
                  int start,
                  int end) {
    // Parcourt chaque colonne du bloc à résoudre
    for (int col = start; col < end; ++col) {
      // Calcul de la solution pour la colonne actuelle en divisant le nouvel élément par le pivot
      X[col] = newB[col - start] / L[col][col];
      // Mise à jour des éléments restants du vecteur résultant pour cette colonne
      for (int line = col + 1; line < end; ++line)
        newB[line - start] -= L[line][col] * X[col];
    }
  }

  // Cette fonction reçoit les solutions précédentes et résout le bloc courant
  void receiveAndSolve(const DistributedRowMatrix &L,
                       const DistributedBlockVector &B,
                       DistributedBlockVector &X,
                       float* newB,
                       float* prevX,
                       OPP::MPI::Ring &ring,
                       std::size_t m) {
    // Boucle à travers tous les processus jusqu'au processus courant
    for (int proc = 0; proc < ring.getRank(); proc++)
    {
      // Réception des solutions précédentes depuis les processus précédents
      ring.Recv(prevX + m * proc, m, MPI_FLOAT);
      // Si ce n'est pas le dernier processus, envoie les solutions au processus suivant
      if (ring.getNext() != 0)
        ring.Send(prevX + m * proc, m, MPI_FLOAT);
      // Mise à jour des éléments du vecteur résultant basée sur les solutions précédentes reçues
      for (int col = proc * m; col < ((proc + 1) * m); col++)
        for (int line = B.Start(); line < B.End(); ++line)
          newB[line - B.Start()] -= L[line][col] * prevX[col];
    }
    // Résolution du bloc local en utilisant les mises à jour des vecteurs résultants
    solveBlock(L, X, newB, B.Start(), B.End());
    // Si ce n'est pas le dernier processus, envoie les résultats au processus suivant
    if (ring.getNext() != 0)
      ring.Send(&X[m * ring.getRank()], m, MPI_FLOAT);
  }
}

// Cette fonction résout un système linéaire distribué Lx = B
void Solve(const OPP::MPI::Communicator &communicator,
           const DistributedRowMatrix &L,
           const DistributedBlockVector &B,
           DistributedBlockVector &X,
           const std::size_t N)
{
  OPP::MPI::Ring ring(communicator.communicator);

  // Calcul de la taille de bloc
  const size_t m = (N + ring.getSize() - 1) / ring.getSize();

  // Vérification de la taille de bloc
  if (m < B.End() - B.Start())
    std::cerr << "Bad value for m=" << m << std::endl;

  // Copie du vecteur B dans un tableau
  float* newB = new float[B.End() - B.Start()];
  for (int i = B.Start(), n = 0; i < B.End(); i++, n++)
    newB[n] = B[i];

  // Si le rang est 0, on résout le système pour le premier bloc
  if (ring.getRank() == 0) {
    solveBlock(L, X, newB, 0, B.End());
    ring.Send(&X[0], m, MPI_FLOAT);
  }
  // Sinon, on reçoit les solutions précédentes et on résout le système pour le bloc courant
  else {
    float* prevX = new float[m * ring.getRank()];
    receiveAndSolve(L, B, X, newB, prevX, ring, m);
  }
}