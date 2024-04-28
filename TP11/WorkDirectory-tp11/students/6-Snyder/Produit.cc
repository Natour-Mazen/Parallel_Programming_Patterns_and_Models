#include <6-Snyder/Produit.h>
#include <OPP/MPI/OPP_MPI.h>
#include <DistributedBlockMatrix.h>
#include <memory>
#include <thread>

namespace {


  /**
   * Transposes data across the MPI Torus.
   *
   * @param torus The torus network object.
   * @param buffer The buffer containing the data to be transposed.
   * @param L The total length of the buffer.
   */
  void transposition(const OPP::MPI::Torus &torus, float *buffer, const int L) {
    using Direction = OPP::MPI::Torus::Direction; // Define alias for Torus direction

    // Get the ranks of the current process in the row and column rings
    const int x = torus.getRowRing().getRank();
    const int y = torus.getColumnRing().getRank();

    // Allocate temporary buffers for data exchange
    auto *buffer1 = new float[L];
    auto *buffer2 = new float[L];

    // Transpose data based on the relative positions of the processes
    if (x < y) {
      // If current process is to the left of the diagonal
      torus.Send(buffer, L, MPI_FLOAT, Direction::EAST); // Send data to the right

      // Loop for receiving and sending data to the right
      for (int i = 0; i < x; ++i) {
        torus.Recv(buffer1, L, MPI_FLOAT, Direction::WEST);
        torus.Send(buffer1, L, MPI_FLOAT, Direction::EAST);
      }

      // Loop for receiving and sending data back from the right
      for (int i = 0; i < x; ++i) {
        torus.Recv(buffer2, L, MPI_FLOAT, Direction::EAST);
        torus.Send(buffer2, L, MPI_FLOAT, Direction::WEST);
      }

      // Receive the final data from the right
      torus.Recv(buffer, L, MPI_FLOAT, Direction::EAST);
    } else if (x > y) {
      // If current process is below the diagonal
      torus.Send(buffer, L, MPI_FLOAT, Direction::SOUTH); // Send data upwards

      // Loop for receiving and sending data upwards
      for (int i = 0; i < y; ++i) {
        torus.Recv(buffer1, L, MPI_FLOAT, Direction::NORTH);
        torus.Send(buffer1, L, MPI_FLOAT, Direction::SOUTH);
      }

      // Loop for receiving and sending data back downwards
      for (int i = 0; i < y; ++i) {
        torus.Recv(buffer2, L, MPI_FLOAT, Direction::SOUTH);
        torus.Send(buffer2, L, MPI_FLOAT, Direction::NORTH);
      }

      // Receive the final data from above
      torus.Recv(buffer, L, MPI_FLOAT, Direction::SOUTH);
    } else {
      // If current process is on the diagonal
      // Loop for receiving and sending data to the left
      for (int i = 0; i < x; ++i) {
        torus.Recv(buffer1, L, MPI_FLOAT, Direction::NORTH);
        torus.Send(buffer1, L, MPI_FLOAT, Direction::WEST);
      }

      // Loop for receiving and sending data upwards
      for (int i = 0; i < x; ++i) {
        torus.Recv(buffer2, L, MPI_FLOAT, Direction::WEST);
        torus.Send(buffer2, L, MPI_FLOAT, Direction::NORTH);
      }
    }

    // Free allocated memory
    delete[] buffer1;
    delete[] buffer2;
  }

  /**
   * Broadcasts and adds row data across the MPI Torus.
   *
   * @param torus The torus network object.
   * @param x The column index of the process that will broadcast the row.
   * @param k The row index to be broadcasted.
   * @param src Pointer to the source buffer that contains the data of the row to be broadcasted.
   * @param dest Pointer to the destination buffer where the broadcasted data will be stored.
   * @param L The total length of the buffer.
   * @param r The width/height of each block of the matrix.
   */
  void BroadcastRowAdd(const OPP::MPI::Torus &torus, const int x, const int k,
                       float *src, float *dest, const int L, const int r) {
    if (torus.getColumnRing().getRank() == x) { // Check if current process is in the specified column
      using Direction = OPP::MPI::Torus::Direction;

      if (torus.getRowRing().getRank() == k) { // Check if current process is in the specified row
        // Broadcast row data to the right and add it to destination buffer
        for (int i = 0; i < r; ++i) {
          torus.Send(&src[i * L / r], L / r, MPI_FLOAT, Direction::EAST);
        }
        for (int i = 0; i < r; ++i) {
          torus.Recv(&dest[i * L / r], L / r, MPI_FLOAT, Direction::WEST);
          for (int j = i * L / r; j < (i + 1) * L / r; ++j) {
            dest[i * L / r + i] += dest[j];
          }
        }
      } else {
        // Receive row data from the left, add it to source buffer, and send to the right
        torus.Recv(dest, L / r, MPI_FLOAT, Direction::WEST);
        for (int i = 0; i < r - 1; ++i) {
          for (int j = i * L / r; j < (i + 1) * L / r; ++j) {
            src[j] += dest[j];
          }
          MPI_Request request =
              torus.AsyncSend(&src[i * L / r], L / r, MPI_FLOAT, Direction::EAST);
          torus.Recv(&dest[(i + 1) * L / r], L / r, MPI_FLOAT, Direction::WEST);
          MPI_Wait(&request, MPI_STATUS_IGNORE);
        }
        for (int j = (r - 1) * L / r; j < r * L / r; ++j) {
          src[j] += dest[j];
        }
        torus.Send(&src[(r - 1) * L / r], L / r, MPI_FLOAT, Direction::EAST);
      }
    }
  }

  /**
   * Rotates data vertically across the MPI Torus.
   *
   * @param torus The torus network object.
   * @param buffer The buffer containing the data to be rotated.
   * @param L The total length of the buffer.
   */
  void RotationVerticale(const OPP::MPI::Torus &torus, float *buffer,
                         const int L) {
    // Rotate data by sending upwards and receiving downwards
    torus.Send(buffer, L, MPI_FLOAT, OPP::MPI::Torus::Direction::NORTH);
    torus.Recv(buffer, L, MPI_FLOAT, OPP::MPI::Torus::Direction::SOUTH);
  }

  /**
   * Initializes buffers with data from DistributedBlockMatrix.
   *
   * @param A The distributed block matrix A.
   * @param B The distributed block matrix B.
   * @param C The distributed block matrix C.
   * @param bufferA Buffer for block matrix A.
   * @param bufferB Buffer for block matrix B.
   * @param r The width/height of each block of the matrix.
   */
  void init(const DistributedBlockMatrix &A, const DistributedBlockMatrix &B,
            DistributedBlockMatrix &C, float *bufferA, float *bufferB,
            const int r) {
    int A_start = A.Start(); // Starting index for block matrix A
    int B_start = B.Start(); // Starting index for block matrix B
    int C_start = C.Start(); // Starting index for block matrix C

    // Copy data from matrices A and B to buffers, and initialize matrix C with zeros
    for (int i = 0; i < r; ++i) {
      for (int j = 0; j < r; ++j) {
        bufferA[i * r + j] = A[i + A_start][j + A[i + A_start].Start()];
        bufferB[i * r + j] = B[j + B_start][i + B[j + B_start].Start()];
        C[i + C_start][j + C[i + C_start].Start()] = 0.0f;
      }
    }
  }

} // namespace

/**
 * Performs matrix multiplication using MPI Torus communication.
 *
 * @param torus The torus network object.
 * @param A The distributed block matrix A.
 * @param B The distributed block matrix B.
 * @param C The distributed block matrix C.
 */
void Produit(const OPP::MPI::Torus &torus, const DistributedBlockMatrix &A,
             const DistributedBlockMatrix &B, DistributedBlockMatrix &C) {
  const int n = sqrt(torus.getCommunicator().size); // Dimension of the MPI grid
  const int x = torus.getColumnRing().getRank(); // Rank of current process in column ring
  const int y = torus.getRowRing().getRank(); // Rank of current process in row ring
  const int r = C.End() - C.Start(); // Size of local block matrix
  const int L = r * r; // Size of data buffer

  auto *bufferA = new float[L]; // Buffer for block matrix A
  auto *bufferB = new float[L]; // Buffer for block matrix B
  auto *send_bufferC = new float[L]; // Buffer for sending data for matrix C
  auto *recv_bufferC = new float[L]; // Buffer for receiving data for matrix C

  // Initialize buffers with data from matrices A and B
  init(A, B, C, bufferA, bufferB, r);

  // Transpose matrix B
  transposition(torus, bufferB, L);

  // Compute initial products and store in send buffer for matrix C
  for (int i = 0; i < L; ++i) {
    send_bufferC[i] = bufferA[i] * bufferB[i];
  }

  // Main loop for matrix multiplication algorithm
  for (int k = 0; k < n; ++k) {
    int diag = (x + k) % n; // Diagonal index for broadcast and receive
    // Broadcast and add row data for the current diagonal
    BroadcastRowAdd(torus, x, diag, send_bufferC, recv_bufferC, L, r);
    if (diag == y) {
      // If current process is on the diagonal, copy received data to matrix C
      for (int i = 0; i < r; ++i) {
        for (int j = 0; j < r; ++j) {
          C[i + C.Start()][j + C[i + C.Start()].Start()] = recv_bufferC[i * r + j];
        }
      }
    }

    // Rotate matrix B vertically
    RotationVerticale(torus, bufferB, L);

    // Recompute products for the next iteration
    for (int i = 0; i < L; ++i) {
      send_bufferC[i] = bufferA[i] * bufferB[i];
    }
  }

  // Free allocated memory
  delete[] bufferA;
  delete[] bufferB;
  delete[] send_bufferC;
  delete[] recv_bufferC;
}
/**********************************/
/*   AL NATOUR MAZEN, M1 Info CL  */
/**********************************/