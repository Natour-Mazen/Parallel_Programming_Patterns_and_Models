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
      const unsigned tidX = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned tidY = blockIdx.y * blockDim.y + threadIdx.y;

      if (tidX < size && tidY < size) {
        const unsigned offset = tidX + tidY * map.imageWidth;

        output[offset] = input[map[offset]];
      }
    }

    template <typename T, typename Functor>
    __host__ void Scatter(OPP::CUDA::DeviceBuffer<T> &dev_input,
                          OPP::CUDA::DeviceBuffer<T> &dev_output,
                          Functor &map)
    {
      // TODO
     /* int blockSize = 256;
      int numBlocks = (dev_input.getNbElements() + blockSize - 1) / blockSize;
      kernelScatter<<<numBlocks, blockSize>>>(dev_input.getDevicePointer(),
                                              dev_output.getDevicePointer(),
                                              map,
                                              dev_input.getNbElements());*/
      const dim3 threads(32, 32);
      const dim3 blocs((map.imageWidth + 32 - 1) / 32,
                       (map.imageHeight + 32 - 1) / 32);

      kernelScatter<<<blocs, threads>>>(
          dev_input.getDevicePointer(),
          dev_output.getDevicePointer(),
          map,
          dev_input.getNbElements()
      );
      cudaDeviceSynchronize();
    }
  } // namespace CUDA
} // namespace OPP

#endif