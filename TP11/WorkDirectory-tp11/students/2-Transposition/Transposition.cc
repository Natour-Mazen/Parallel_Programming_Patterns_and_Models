#include <2-Transposition/Transposition.h>
#include <OPP/MPI/OPP_MPI.h>
#include <utils/DistributedBlockMatrix.h>

#include <thread>
#include <memory>
#include <vector>

namespace {
  // chargement et translation du bloc
  void loadAndTranslate(std::vector<float> &block,
                        const DistributedBlockMatrix &M,
                        const unsigned width)
  {
    // TODO
    for (int i = M.Start(); i < M.End(); ++i)
    {
      for (int j = M[i].Start(); j < M[i].End(); ++j)
      {
        block[(i - M.Start()) + width * (j - M[i].Start())] = M[i][j];
      }
    }
  }

  // sens Lower vers Up (du bas vers le haut)
  void below2above(const OPP::MPI::Torus &torus,
                   const int bSize,
                   const std::vector<float> &block,
                   std::vector<float> &transpose)
  {
    using Direction = OPP::MPI::Torus::Direction;
    const auto row = torus.getRowRing().getRank();
    const auto col = torus.getColumnRing().getRank();
    std::vector<float> buffer(bSize);
    if (row < col) // sous la diagonale : on envoie de gauche à droite
    {
      // TODO
      torus.Send(&block[0], bSize, MPI_FLOAT, Direction::EAST);

      for (int i = 0; i < row; ++i) {
        torus.Recv(&buffer[0], bSize, MPI_FLOAT, Direction::WEST);
        torus.Send(&buffer[0], bSize, MPI_FLOAT, Direction::EAST);
      }
    }
    else if (row > col) // sur la diagonale : on reçoit de bas en haut
    {
      // TODO
      torus.Recv(&transpose[0], bSize, MPI_FLOAT, Direction::SOUTH);

      for (int i = 0; i < col; ++i) {
        torus.Recv(&buffer[0], bSize, MPI_FLOAT, Direction::SOUTH);
        torus.Send(&buffer[0], bSize, MPI_FLOAT, Direction::NORTH);
      }
    }
    else // sur la diagonale
    {
      // TODO
      for (int i = 0; i < row; ++i) {
        torus.Recv(&buffer[0], bSize, MPI_FLOAT, Direction::WEST);
        torus.Send(&buffer[0], bSize, MPI_FLOAT, Direction::NORTH);
      }
    }
  }

  // sens Up vers Lower (du haut vers le bas)
  void above2below(const OPP::MPI::Torus &torus,
                   const int bSize,
                   const std::vector<float> &block,
                   std::vector<float> &transpose)
  {
    // TODO
    using Direction = OPP::MPI::Torus::Direction;

    const int row = torus.getRowRing().getRank();
    const int col = torus.getColumnRing().getRank();

    std::vector<float> buffer(bSize);

    if (row < col) {
      torus.Recv(&transpose[0], bSize, MPI_FLOAT, Direction::EAST);

      for (int i = 0; i < row; ++i) {
        torus.Recv(&buffer[0], bSize, MPI_FLOAT, Direction::EAST);
        torus.Send(&buffer[0], bSize, MPI_FLOAT, Direction::WEST);
      }
    } else if (row > col) {
      torus.Send(&block[0], bSize, MPI_FLOAT, Direction::SOUTH);

      for (int i = 0; i < col; ++i) {
        torus.Recv(&buffer[0], bSize, MPI_FLOAT, Direction::NORTH);
        torus.Send(&buffer[0], bSize, MPI_FLOAT, Direction::SOUTH);
      }
    } else {
      for (int i = 0; i < row; ++i) {
        torus.Recv(&buffer[0], bSize, MPI_FLOAT, Direction::NORTH);
        torus.Send(&buffer[0], bSize, MPI_FLOAT, Direction::WEST);
      }
    }
  }

  // sauvegarde du résultat
  void saveBlock(const std::vector<float> &transpose,
                 DistributedBlockMatrix &M,
                 const unsigned width)
  {
    // TODO
    for (int i = M.Start(); i < M.End(); ++i)
      for (int j = M[i].Start(); j < M[i].End(); ++j)
        M[i][j] = transpose[(i - M.Start()) * width + (j - M[i].Start())];
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
  std::vector<float> block(bSize);
  std::vector<float> transpose(bSize);
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
