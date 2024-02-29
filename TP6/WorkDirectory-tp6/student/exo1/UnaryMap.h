#ifndef UNARY_MAP_H
#define UNARY_MAP_H

#include <OPP_cuda.cuh>

namespace OPP {
  namespace CUDA {

    /**
     * @brief Cuda kernel doing the Unary map for every i=0 to size-1
     *
     * @tparam Tsrc datatype of the source
     * @tparam Tdst datatype of the destination
     * @tparam Functor datatype of the transformation
     * @param src source array of data
     * @param dst destination array of data
     * @param size size of source and destination arrays
     * @param functor transformation from source to destination
     */
    template <typename Tsrc, typename TDest, typename FunctorType>
    __global__ void unary_map(const Tsrc src,
                              TDest dest,
                              const std::size_t size,
                              const FunctorType functor)
    {
      // TODO
        const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < size) {
            dest[idx] = functor(src[idx]);
        }
    }

    /**
     * @brief Apply an unary map
     *
     * The unary map does dst[i] = functor(src[i]) for i=0 to size-1.
     *
     * @tparam Tsrc datatype of the source
     * @tparam Tdst datatype of the destination
     * @tparam Functor datatype of the transformation
     * @param src source array of data
     * @param dst destination array of data
     * @param size size of source and destination arrays
     * @param functor transformation from source to destination
     */
    template <typename Tsrc, typename Tdst, typename Functor>
    void UnaryMap(const Tsrc src, Tdst dst, const std::size_t size, const Functor &functor)
    {
      // Attention : ici la taille des vecteurs n'est pas toujours un multiple du nombre de threads.
      // Il faut donc corriger l'exemple du cours... avec un garde-fou !
      
      // TODO
         constexpr int threadsPerBlock = 256;
          int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

        unary_map<<<blocksPerGrid, threadsPerBlock>>>(src, dst, size, functor);
    }
  } // namespace CUDA
} // namespace OPP

#endif