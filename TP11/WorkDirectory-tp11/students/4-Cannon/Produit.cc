#include <4-Cannon/Produit.h>
#include <OPP/MPI/OPP_MPI.h>

#include <DistributedBlockMatrix.h>
#include <memory>

namespace {

  // Rotate the buffer horizontally among processes
  void RotationHorizontale(const OPP::MPI::Torus &torus, float *buffer,
                           const int L) {
    // Send the buffer to the left and receive from the right
    torus.Send(buffer, L, MPI_FLOAT, OPP::MPI::Torus::Direction::WEST);
    torus.Recv(buffer, L, MPI_FLOAT, OPP::MPI::Torus::Direction::EAST);
  }

  // Rotate the buffer vertically among processes
  void RotationVerticale(const OPP::MPI::Torus &torus, float *buffer,
                         const int L) {
    // Send the buffer upwards and receive from below
    torus.Send(buffer, L, MPI_FLOAT, OPP::MPI::Torus::Direction::NORTH);
    torus.Recv(buffer, L, MPI_FLOAT, OPP::MPI::Torus::Direction::SOUTH);
  }

  // Sequential matrix multiplication of two square matrices A and B.
  // @param A left operand
  // @param B right operand
  // @param C result matrix (out)
  // @param n width/height of the matrices
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
  const int r = C.End() - C.Start();
  const int L = r * r;

  // Allocate memory for buffers
  auto *bufferA = new float[L];
  auto *bufferB = new float[L];

  // Initialize buffers and result matrix
  init(A, B, C, bufferA, bufferB, r);

  // Rotate buffers to align data for initial multiplication
  RotationHorizontale(torus, bufferA, L);
  RotationVerticale(torus, bufferB, L);

  // Perform matrix multiplication iteratively
  for (int k = 0; k < n; ++k) {
    ProduitSequentiel(bufferA, bufferB, C, r);

    // Rotate buffers for the next iteration
    RotationHorizontale(torus, bufferA, L);
    RotationVerticale(torus, bufferB, L);
  }

  // Clean up allocated memory
  delete[] bufferA;
  delete[] bufferB;
}
/**********************************/
/*   AL NATOUR MAZEN, M1 Info CL  */
/**********************************/
