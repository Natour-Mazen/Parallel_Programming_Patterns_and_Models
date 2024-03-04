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
      const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

      if (tid < size) {
        output[map[tid]] = input[tid];
      }
    }

    template <typename T, typename Functor>
    __host__ void Scatter(OPP::CUDA::DeviceBuffer<T> &dev_input,
                          OPP::CUDA::DeviceBuffer<T> &dev_output,
                          Functor &map)
    {
      const size_t size = dev_input.getNbElements();
      const dim3 threads(1024);
      const dim3 blocs = (size + threads.x - 1) / threads.x;

      kernelScatter<<<blocs, threads>>>(
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