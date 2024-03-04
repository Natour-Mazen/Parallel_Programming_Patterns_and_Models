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
      const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

      if (tid < size) {
        output[tid] = input[map[tid]];
      }
    }

    template <typename T, typename Functor>
    void Gather(OPP::CUDA::DeviceBuffer<T> &dev_input,
                OPP::CUDA::DeviceBuffer<T> &dev_output,
                Functor &map)
    {
      const size_t size = dev_input.getNbElements();
      const dim3 threads(1024);
      const dim3 blocs = (size + threads.x - 1) / threads.x;

      kernelGather<<<blocs, threads>>>(
          dev_input.getDevicePointer(),
          dev_output.getDevicePointer(),
          map,
          size
      );
      cudaDeviceSynchronize();
    }

  } // namespace CUDA
} // namespace OPP

/**********************************/
/*   AL NATOUR MAZEN, M1 Info CL  */
/**********************************/

#endif
