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
      const unsigned tidX = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned tidY = blockIdx.y * blockDim.y + threadIdx.y;

      if (tidX < size && tidY < size) {
        const unsigned offset = tidX + tidY * map.imageWidth;

        output[offset] = input[map[offset]];
      }
    }

    template <typename T, typename Functor>
    void Gather(OPP::CUDA::DeviceBuffer<T> &dev_input,
                OPP::CUDA::DeviceBuffer<T> &dev_output,
                Functor &map)
    {
      const dim3 threads(32, 32);
      const dim3 blocs((map.imageWidth + 32 - 1) / 32,
                       (map.imageHeight + 32 - 1) / 32);

      kernelGather<<<blocs, threads>>>(
          dev_input.getDevicePointer(),
          dev_output.getDevicePointer(),
          map,
          dev_input.getNbElements()
      );
    }

  } // namespace CUDA
} // namespace OPP
#endif
