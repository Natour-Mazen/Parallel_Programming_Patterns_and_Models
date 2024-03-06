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
      // Calculate the global thread ID
      const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
      // Check if the thread ID is within the size of the array
      if (tid < size) {
        // Gather the element from the input array based on the map and store it in the output array
        output[tid] = input[map[tid]];
      }
    }

    template <typename T, typename Functor>
    void Gather(OPP::CUDA::DeviceBuffer<T> &dev_input,
                OPP::CUDA::DeviceBuffer<T> &dev_output,
                Functor &map)
    {
      // Get the number of elements in the input array
      const size_t size = dev_input.getNbElements();
      // Define the number of threads per block
      const dim3 threads(1024);
      // Calculate the number of blocks needed to cover all elements
      const dim3 blocs = (size + threads.x - 1) / threads.x;

      // Launch the gather kernel
      kernelGather<<<blocs, threads>>>(
          dev_input.getDevicePointer(),
          dev_output.getDevicePointer(),
          map,
          size
      );
      // Synchronize the device to ensure all threads have finished before proceeding
      cudaDeviceSynchronize();
    }

  } // namespace CUDA
} // namespace OPP

/**********************************/
/*   AL NATOUR MAZEN, M1 Info CL  */
/**********************************/

#endif
