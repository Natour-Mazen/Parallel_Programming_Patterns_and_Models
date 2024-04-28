#include <5-Fox/Produit.h>
#include <OPP/MPI/OPP_MPI.h>

#include <DistributedBlockMatrix.h>
#include <memory>

namespace {

  // Broadcasts rows of the matrix A among processes in a row
  void BroadcastRow(const OPP::MPI::Torus &torus, const int x, const int k,
                    float *src, float *dest, const int L, const int r) {
    if (torus.getColumnRing().getRank() == x) {
      using Direction = OPP::MPI::Torus::Direction;

      // If the current process is the owner of the row, broadcast to the right
      if (torus.getRowRing().getRank() == k) {
        for (int i = 0; i < r; ++i) {
          torus.Send(&src[i * L / r], L / r, MPI_FLOAT, Direction::EAST);
          for (int j = 0; j < L / r; ++j) {
            dest[i * L / r + j] = src[i * L / r + j];
          }
        }
      }
      // If the current process is the next to the owner of the row, receive from the left
      else if (torus.getRowRing().getNext() == k) {
        for (int i = 0; i < r; ++i) {
          torus.Recv(&dest[i * L / r], L / r, MPI_FLOAT, Direction::WEST);
        }
      }
      // If neither owner nor next, receive from the left and send to the right
      else {
        torus.Recv(dest, L / r, MPI_FLOAT, Direction::WEST);
        for (int i = 0; i < r - 1; ++i) {
          MPI_Request request = torus.AsyncSend(&src[i * L / r], L / r, MPI_FLOAT, Direction::EAST);
          torus.Recv(&dest[(i + 1) * L / r], L / r, MPI_FLOAT, Direction::WEST);
          MPI_Wait(&request, MPI_STATUS_IGNORE);
        }
        torus.Send(&dest[(r - 1) * L / r], L / r, MPI_FLOAT, Direction::EAST);
      }
    }
  }

  // Rotate the buffer vertically among processes
  void RotationVerticale(const OPP::MPI::Torus &torus, float *buffer,
                         const int L) {
    // Send the buffer upwards and receive from below
    torus.Send(buffer, L, MPI_FLOAT, OPP::MPI::Torus::Direction::NORTH);
    torus.Recv(buffer, L, MPI_FLOAT, OPP::MPI::Torus::Direction::SOUTH);
  }

  // Sequential matrix multiplication of two square matrices A and B
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
    for (int i = A.Start(); i < A.End(); ++i) {
      for (int j = A[i].Start(); j < A[i].End(); ++j) {
        bufferA[(j - A[i].Start()) + r * (i - A.Start())] = A[i][j];
      }
    }

    for (int i = B.Start(); i < B.End(); ++i) {
      for (int j = B[i].Start(); j < B[i].End(); ++j) {
        bufferB[(j - B[i].Start()) + r * (i - B.Start())] = B[i][j];
      }
    }

    for (int i = C.Start(); i < C.End(); ++i) {
      for (int j = C[i].Start(); j < C[i].End(); ++j) {
        C[i][j] = 0.0f;
      }
    }
  }
} // namespace

// Distributed matrix multiplication function using Fox's algorithm
void Produit(const OPP::MPI::Torus &torus,
             const DistributedBlockMatrix &A,
             const DistributedBlockMatrix &B,
             DistributedBlockMatrix &C)
{
  // Calculate necessary parameters
  const int n = sqrt(torus.getCommunicator().size);
  const int x = torus.getColumnRing().getRank();
  const int r = C.End() - C.Start(); // Width/height of each block
  const int L = r * r; // Total size of each block

  // Allocate memory for send/receive buffers
  auto *send_bufferA = new float[L];
  auto *send_bufferB = new float[L];

  // Initialize buffers and result matrix
  init(A, B, C, send_bufferA, send_bufferB, r);

  // Allocate memory for receiving buffer
  auto *recv_bufferA = new float[L];

  // Perform matrix multiplication using Fox's algorithm
  for (int k = 0; k < n; ++k) {
    // Broadcast rows of A among processes in the same row
    BroadcastRow(torus, x, (x + k) % n, send_bufferA, recv_bufferA, L, r);

    // Perform sequential multiplication with the received rows and bufferB
    ProduitSequentiel(recv_bufferA, send_bufferB, C, r);

    // Rotate bufferB vertically for the next iteration
    RotationVerticale(torus, send_bufferB, L);
  }

  // Clean up allocated memory
  delete[] send_bufferA;
  delete[] send_bufferB;
  delete[] recv_bufferA;
}
/**********************************/
/*   AL NATOUR MAZEN, M1 Info CL  */
/**********************************/