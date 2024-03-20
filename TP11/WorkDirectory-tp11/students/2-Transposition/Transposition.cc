#include <2-Transposition/Transposition.h>
#include <MPI/OPP_MPI.h>
#include <utils/DistributedBlockMatrix.h>

#include <memory>
#include <thread>


namespace {
  // chargement et translation du bloc
  void loadAndTranslate(std::shared_ptr<float> &block,
                        const DistributedBlockMatrix &M,
                        const unsigned width)
  {
    // TODO
  }

  // sens Lower vers Up (du bas vers le haut)
  void below2above(const OPP::MPI::Torus &torus,
                   const int bSize,
                   const std::shared_ptr<float> &block,
                   std::shared_ptr<float> &transpose)
  {
    using Direction = OPP::MPI::Torus::Direction;
    const auto row = torus.getRowRing().getRank();
    const auto col = torus.getColumnRing().getRank();
    std::unique_ptr<float> buffer(new float[bSize]);
    if (row < col) // sous la diagonale : on envoie de gauche à droite
    {
      // TODO
    }
    else if (row > col) // sur la diagonale : on reçoit de bas en haut
    {
      // TODO
    }
    else // sur la diagonale
    {
      // TODO
    }
  }

  // sens Up vers Lower (du haut vers le bas)
  void above2below(const OPP::MPI::Torus &torus,
                   const int bSize,
                   const std::shared_ptr<float> &block,
                   std::shared_ptr<float> &transpose)
  {
    // TODO
  }

  // sauvegarde du résultat
  void saveBlock(const std::shared_ptr<float> &transpose,
                 DistributedBlockMatrix &M,
                 const unsigned width)
  {
    // TODO
  }
} // namespace

void Transposition(const OPP::MPI::Torus &torus,
                   const DistributedBlockMatrix &A,
                   DistributedBlockMatrix &B,
                   const int N, // width and height of matrices A and B
                   const int P  // width and height of the processes grid
)
{
  // position dans la grille
  const auto x = torus.getRowRing().getRank();
  const auto y = torus.getColumnRing().getRank();

  // information sur les blocs
  const unsigned height = (N + P - 1) / P;
  const unsigned width = (N + P - 1) / P;
  const unsigned bSize = height * width;

  // charger le bloc & le transposer
  std::shared_ptr<float> block(new float[bSize]);
  std::shared_ptr<float> transpose(new float[bSize]);
  if (x == y) // attention au cas de la diagonale ... il faut copier le résultat !
    loadAndTranslate(transpose, A, width);
  else
    loadAndTranslate(block, A, width);

  // on traite chaque sens en parallèle :
  {
    // on envoie (sauf sur diagonal), ensuite on sert de relais et cela dans chaque sens
    std::thread thread = std::thread([&]() { above2below(torus, bSize, block, transpose); });
    below2above(torus, bSize, block, transpose);
    thread.join();
  }

  // ne reste plus qu'à sauvegarder dans la matrice distribuée
  saveBlock(transpose, B, width);

  // that's all, folks!
}
