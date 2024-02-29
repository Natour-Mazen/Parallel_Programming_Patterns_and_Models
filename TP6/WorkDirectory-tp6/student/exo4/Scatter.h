#ifndef SCATTER_H
#define SCATTER_H

#include <OPP_cuda.cuh>

namespace OPP {
  namespace CUDA {
    template <typename T, typename Functor>
    __global__ void kernelScatter(T const *const input,
                                  T *const output,
                                  Functor map,
                                  const size_t size)
    {
      // TODO
    }

    template <typename T, typename Functor>
    __host__ void Scatter(OPP::CUDA::DeviceBuffer<T> &dev_input,
                          OPP::CUDA::DeviceBuffer<T> &dev_output,
                          Functor &map)
    {
      // TODO
    }
  } // namespace CUDA
} // namespace OPP

#endif