#ifndef GATHER_H
#define GATHER_H

#include <OPP_cuda.cuh>

namespace OPP {
  namespace CUDA {
    template <typename T, typename Functor>
    __global__ void kernelGather(T const *const input,
                                 T *const output,
                                 Functor map,
                                 const size_t size)
    {
      // TODO
    }

    template <typename T, typename Functor>
    void Gather(OPP::CUDA::DeviceBuffer<T> &dev_input,
                OPP::CUDA::DeviceBuffer<T> &dev_output,
                Functor &map)
    {
      // TODO
    }

  } // namespace CUDA
} // namespace OPP
#endif
