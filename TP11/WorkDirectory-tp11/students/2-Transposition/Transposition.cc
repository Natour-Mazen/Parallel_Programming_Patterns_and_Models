#include <2-Transposition/Transposition.h>
#include <OPP/MPI/OPP_MPI.h>
#include <utils/DistributedBlockMatrix.h>

#include <thread>
#include <memory>
#include <vector>

namespace {
  // Load and translate the block
  void loadAndTranslate(std::vector<float> &block,
                        const DistributedBlockMatrix &M,
                        const unsigned width)
  {
    // Load each element of the block from the DistributedBlockMatrix M
    for (int i = M.Start(); i < M.End(); ++i)
    {
      for (int j = M[i].Start(); j < M[i].End(); ++j)
      {
        // Translate the loaded elements to fit into a linear vector 'block'
        block[(i - M.Start()) + width * (j - M[i].Start())] = M[i][j];
      }
    }
  }

  // Transpose from Lower to Upper (from bottom to top)
  void below2above(const OPP::MPI::Torus &torus,
                   const int bSize,
                   const std::vector<float> &block,
                   std::vector<float> &transpose)
  {
    using Direction = OPP::MPI::Torus::Direction;
    const auto row = torus.getRowRing().getRank(); // Get the row index in the Torus grid
    const auto col = torus.getColumnRing().getRank(); // Get the column index in the Torus grid
    std::vector<float> buffer(bSize); // Create a buffer to store received data temporarily

    if (row < col) // Below the diagonal: send data from left to right
    {
      torus.Send(&block[0], bSize, MPI_FLOAT, Direction::EAST); // Send the block data to the east neighbor

      // Relay the data to the east neighbors until reaching the last row
      for (int i = 0; i < row; ++i) {
        torus.Recv(&buffer[0], bSize, MPI_FLOAT, Direction::WEST); // Receive data from the west neighbor
        torus.Send(&buffer[0], bSize, MPI_FLOAT, Direction::EAST); // Send data to the east neighbor
      }
    }
    else if (row > col) // On the diagonal: receive data from bottom to top
    {
      torus.Recv(&transpose[0], bSize, MPI_FLOAT, Direction::SOUTH); // Receive the transposed data from the south neighbor

      // Relay the data to the north neighbors until reaching the last column
      for (int i = 0; i < col; ++i) {
        torus.Recv(&buffer[0], bSize, MPI_FLOAT, Direction::SOUTH); // Receive data from the south neighbor
        torus.Send(&buffer[0], bSize, MPI_FLOAT, Direction::NORTH); // Send data to the north neighbor
      }
    }
    else // On the diagonal
    {
      // Relay the data to the north neighbors until reaching the last row
      for (int i = 0; i < row; ++i) {
        torus.Recv(&buffer[0], bSize, MPI_FLOAT, Direction::WEST); // Receive data from the west neighbor
        torus.Send(&buffer[0], bSize, MPI_FLOAT, Direction::NORTH); // Send data to the north neighbor
      }
    }
  }

  // Transpose from Upper to Lower (from top to bottom)
  void above2below(const OPP::MPI::Torus &torus,
                   const int bSize,
                   const std::vector<float> &block,
                   std::vector<float> &transpose)
  {
    using Direction = OPP::MPI::Torus::Direction;

    const int row = torus.getRowRing().getRank(); // Get the row index in the Torus grid
    const int col = torus.getColumnRing().getRank(); // Get the column index in the Torus grid

    std::vector<float> buffer(bSize); // Create a buffer to store received data temporarily

    if (row < col) {
      torus.Recv(&transpose[0], bSize, MPI_FLOAT, Direction::EAST); // Receive the transposed data from the east neighbor

      // Relay the data to the west neighbors until reaching the last row
      for (int i = 0; i < row; ++i) {
        torus.Recv(&buffer[0], bSize, MPI_FLOAT, Direction::EAST); // Receive data from the east neighbor
        torus.Send(&buffer[0], bSize, MPI_FLOAT, Direction::WEST); // Send data to the west neighbor
      }
    } else if (row > col) {
      torus.Send(&block[0], bSize, MPI_FLOAT, Direction::SOUTH); // Send the block data to the south neighbor

      // Relay the data to the south neighbors until reaching the last column
      for (int i = 0; i < col; ++i) {
        torus.Recv(&buffer[0], bSize, MPI_FLOAT, Direction::NORTH); // Receive data from the north neighbor
        torus.Send(&buffer[0], bSize, MPI_FLOAT, Direction::SOUTH); // Send data to the south neighbor
      }
    } else {
      // Relay the data to the west neighbors until reaching the last row
      for (int i = 0; i < row; ++i) {
        torus.Recv(&buffer[0], bSize, MPI_FLOAT, Direction::NORTH); // Receive data from the north neighbor
        torus.Send(&buffer[0], bSize, MPI_FLOAT, Direction::WEST); // Send data to the west neighbor
      }
    }
  }

  // Save the result back to the DistributedBlockMatrix
  void saveBlock(const std::vector<float> &transpose,
                 DistributedBlockMatrix &M,
                 const unsigned width)
  {
    // Save the transposed block back to the DistributedBlockMatrix M
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
/**********************************/
/*   AL NATOUR MAZEN, M1 Info CL  */
/**********************************/