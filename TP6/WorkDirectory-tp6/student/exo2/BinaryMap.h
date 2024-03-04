#ifndef STUDENT_EXO2_BINARY_MAP_H
#define STUDENT_EXO2_BINARY_MAP_H
#include <OPP_cuda.cuh>

namespace OPP::CUDA {

  template <typename TLeft, typename TRight, typename TDest, typename FunctorType>
  __global__ void binary_map(TLeft a, TRight b, TDest result, std::size_t size, FunctorType functor)
  {
    // TODO
    std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
      result[index] = functor(a[index], b[index]);
    }
  }

  // Attention : ici la taille des vecteurs n'est pas toujours un multiple du nombre de threads !
  // Il faut donc corriger l'exemple du cours ...
  template <typename TLeft, typename TRight, typename TDest, typename Functor>
  void BinaryMap(TLeft dev_a, TRight dev_b, TDest dev_result, std::size_t size, Functor &functor)
  {
    // TODO
    int blockSize;
    int minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, binary_map<TLeft, TRight,TDest, Functor>, 0, size);

    const size_t gridSize = (size + blockSize - 1) / blockSize; // This ensures we have enough blocks to cover all elements
    binary_map<<<gridSize, blockSize>>>(dev_a, dev_b, dev_result, size, functor);
    cudaDeviceSynchronize(); // Wait for the kernel to finish
  }

} // namespace OPP::CUDA

/**********************************/
/*   AL NATOUR MAZEN, M1 Info CL  */
/**********************************/
#endif