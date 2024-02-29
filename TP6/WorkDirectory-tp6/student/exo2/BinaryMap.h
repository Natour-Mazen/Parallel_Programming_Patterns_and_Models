#ifndef STUDENT_EXO2_BINARY_MAP_H
#define STUDENT_EXO2_BINARY_MAP_H
#include <OPP_cuda.cuh>

namespace OPP::CUDA {

  template <typename TLeft, typename TRight, typename TDest, typename FunctorType>
  __global__ void binary_map(TLeft a, TRight b, TDest result, std::size_t size, FunctorType functor)
  {
    // TODO
  }

  // Attention : ici la taille des vecteurs n'est pas toujours un multiple du nombre de threads !
  // Il faut donc corriger l'exemple du cours ...
  template <typename TLeft, typename TRight, typename TDest, typename Functor>
  void BinaryMap(TLeft dev_a, TRight dev_b, TDest dev_result, std::size_t size, Functor &functor)
  {
    // TODO
  }

} // namespace OPP::CUDA
#endif