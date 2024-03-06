#ifndef STUDENT_EXO2_BINARY_MAP_H
#define STUDENT_EXO2_BINARY_MAP_H
#include <OPP_cuda.cuh>

namespace OPP::CUDA {

  template <typename TLeft, typename TRight, typename TDest, typename FunctorType>
  __global__ void binary_map(TLeft a, TRight b, TDest result, std::size_t size, FunctorType functor)
  {
    // Calculate the global index for the current thread
    std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    // Check if the index is within the size of the arrays
    if (index < size) {
      // Apply the binary operation to the elements at the current index
      result[index] = functor(a[index], b[index]);
    }
  }

  // Attention : ici la taille des vecteurs n'est pas toujours un multiple du nombre de threads !
  // Il faut donc corriger l'exemple du cours ...
  template <typename TLeft, typename TRight, typename TDest, typename Functor>
  void BinaryMap(TLeft dev_a, TRight dev_b, TDest dev_result, std::size_t size, Functor &functor)
  {
    int blockSize;
    int minGridSize;
    // Determine the block size and grid size using the CUDA Occupancy API
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, binary_map<TLeft, TRight,TDest, Functor>, 0, size);

    // Calculate the grid size to ensure all elements are covered
    const size_t gridSize = (size + blockSize - 1) / blockSize;
    // Launch the kernel with the calculated grid and block size
    binary_map<<<gridSize, blockSize>>>(dev_a, dev_b, dev_result, size, functor);
    // Synchronize the device to ensure all threads have finished before proceeding
    cudaDeviceSynchronize();
  }

} // namespace OPP::CUDA

/**********************************/
/*   AL NATOUR MAZEN, M1 Info CL  */
/**********************************/
#endif