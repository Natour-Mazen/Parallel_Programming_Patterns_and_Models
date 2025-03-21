#include <3-Produit/Produit.h>
#include <OPP/MPI/OPP_MPI.h>
#include <utils/DistributedBlockMatrix.h>

#include <memory>

namespace {

  // Broadcast a row of a matrix among processes in the same column
  void BroadcastRow(const OPP::MPI::Torus &torus, const int x, const int k,
                    float *src, float *dest, const int L, const int r) {
    // Broadcast only if the current process is in the same column as the source
    if (torus.getColumnRing().getRank() == x) {
      using Direction = OPP::MPI::Torus::Direction;

      // If the current process is the source row, send the row to the right
      if (torus.getRowRing().getRank() == k) {
        for (int i = 0; i < r; ++i) {
          torus.Send(&src[i * L / r], L / r, MPI_FLOAT, Direction::EAST);
          // Also copy the row to the destination buffer
          for (int j = 0; j < L / r; ++j) {
            dest[i * L / r + j] = src[i * L / r + j];
          }
        }
      }
      // If the current process is the destination row, receive the row from the left
      else if (torus.getRowRing().getNext() == k) {
        for (int i = 0; i < r; ++i) {
          torus.Recv(&dest[i * L / r], L / r, MPI_FLOAT, Direction::WEST);
        }
      }
      // Otherwise, relay the row from left to right
      else {
        torus.Recv(dest, L / r, MPI_FLOAT, Direction::WEST);
        for (int i = 0; i < r - 1; ++i) {
          MPI_Request request =
              torus.AsyncSend(&src[i * L / r], L / r, MPI_FLOAT, Direction::EAST);
          torus.Recv(&dest[(i + 1) * L / r], L / r, MPI_FLOAT, Direction::WEST);
          MPI_Wait(&request, MPI_STATUS_IGNORE);
        }
        torus.Send(&src[(r - 1) * L / r], L / r, MPI_FLOAT, Direction::EAST);
      }
    }
  }

  // Broadcast a column of a matrix among processes in the same row
  void BroadcastCol(const OPP::MPI::Torus &torus, const int k, const int y,
                    float *src, float *dest, const int L, const int r) {
    // Broadcast only if the current process is in the same row as the source
    if (torus.getRowRing().getRank() == y) {
      using Direction = OPP::MPI::Torus::Direction;

      // If the current process is the source column, send the column downward
      if (torus.getColumnRing().getRank() == k) {
        for (int i = 0; i < r; ++i) {
          torus.Send(&src[i * L / r], L / r, MPI_FLOAT, Direction::SOUTH);
          // Also copy the column to the destination buffer
          for (int j = 0; j < L / r; ++j) {
            dest[i * L / r + j] = src[i * L / r + j];
          }
        }
      }
      // If the current process is the destination column, receive the column from above
      else if (torus.getColumnRing().getNext() == k) {
        for (int i = 0; i < r; ++i) {
          torus.Recv(&dest[i * L / r], L / r, MPI_FLOAT, Direction::NORTH);
        }
      }
      // Otherwise, relay the column from above to below
      else {
        torus.Recv(dest, L / r, MPI_FLOAT, Direction::NORTH);
        for (int i = 0; i < r - 1; ++i) {
          MPI_Request request = torus.AsyncSend(&src[i * L / r], L / r, MPI_FLOAT,
                                                Direction::SOUTH);
          torus.Recv(&dest[(i + 1) * L / r], L / r, MPI_FLOAT, Direction::NORTH);
          MPI_Wait(&request, MPI_STATUS_IGNORE);
        }
        torus.Send(&src[(r - 1) * L / r], L / r, MPI_FLOAT, Direction::SOUTH);
      }
    }
  }

  // Sequential matrix multiplication of two square matrices A and B.
  // @param A left operand
  // @param B right operand
  // @param C result matrix (out)
  // @param r width/height of the matrices
  void ProduitSequentiel(const float *A, const float *B, DistributedBlockMatrix &C, int r) {
    for (int row = C.Start(); row < C.End(); ++row) {
      for (int col = C[row].Start(); col < C[row].End(); ++col) {
        float dot = 0.0;
        for (int k = 0; k < r; ++k) {
          dot += A[k + (row - C.Start()) * r] * B[(col - C[row].Start()) + k * r];
        }
        C[row][col] += dot;
      }
    }
  }

  // Initialize the buffers and the result matrix
  void init(const DistributedBlockMatrix &A, const DistributedBlockMatrix &B,
            DistributedBlockMatrix &C, float *bufferA, float *bufferB,
            const int r) {
    // Initialize bufferA with the values of matrix A
    for (int i = A.Start(); i < A.End(); ++i) {
      for (int j = A[i].Start(); j < A[i].End(); ++j) {
        bufferA[(j - A[i].Start()) + r * (i - A.Start())] = A[i][j];
      }
    }

    // Initialize bufferB with the values of matrix B
    for (int i = B.Start(); i < B.End(); ++i) {
      for (int j = B[i].Start(); j < B[i].End(); ++j) {
        bufferB[(j - B[i].Start()) + r * (i - B.Start())] = B[i][j];
      }
    }

    // Initialize the result matrix C with zeros
    for (int i = C.Start(); i < C.End(); ++i) {
      for (int j = C[i].Start(); j < C[i].End(); ++j) {
        C[i][j] = 0.0f;
      }
    }
  }

} // namespace

// Distributed matrix multiplication function
void Produit(const OPP::MPI::Torus &torus,
             const DistributedBlockMatrix &A,
             const DistributedBlockMatrix &B,
             DistributedBlockMatrix &C)
{
  // Calculate necessary parameters
  const int n = sqrt(torus.getCommunicator().size);
  const int x = torus.getColumnRing().getRank();
  const int y = torus.getRowRing().getRank();
  const int r = C.End() - C.Start(); // Width/height of each block
  const int L = r * r; // Total size of each block

  // Allocate memory for send/receive buffers
  auto *send_bufferA = new float[L];
  auto *send_bufferB = new float[L];

  // Initialize buffers and result matrix
  init(A, B, C, send_bufferA, send_bufferB, r);

  auto *recv_bufferA = new float[L];
  auto *recv_bufferB = new float[L];

  // Broadcast rows and columns iteratively and perform sequential multiplication
  for (int k = 0; k < n; ++k) {
    BroadcastRow(torus, x, k, send_bufferA, recv_bufferA, L, r);
    BroadcastCol(torus, k, y, send_bufferB, recv_bufferB, L, r);

    ProduitSequentiel(send_bufferA, send_bufferB, C, r);
  }

  // Clean up allocated memory
  delete[] send_bufferA;
  delete[] send_bufferB;
  delete[] recv_bufferA;
  delete[] recv_bufferB;
}
/**********************************/
/*   AL NATOUR MAZEN, M1 Info CL  */
/**********************************/
