#include <iostream>
#include <exo4/student.h>
#include <OPP_cuda.cuh>

namespace 
{
  using uchar = unsigned char;

  // CUDA kernel for performing a transformation based on the provided values and repartition
  __global__
      void transformation_kernel(
          const float* const value,
          const unsigned* const repartition,
          float* const transformation,
          const unsigned size
      ){
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x; // Calculate thread ID
    if (tid < size) {
      const uchar xi = uchar(value[tid]); // Convert value to unsigned char
      // Perform transformation based on repartition and store result in transformation array
      transformation[tid] = (255.f *  static_cast<float>(repartition[xi])) / static_cast<float>(size);
    }
  }
}

void StudentWorkImpl::run_Transformation(
	OPP::CUDA::DeviceBuffer<float>& dev_Value,
	OPP::CUDA::DeviceBuffer<unsigned>& dev_repartition,
	OPP::CUDA::DeviceBuffer<float>& dev_transformation // or "transformed"
) {
  const unsigned nbThreads = 1024;

  const unsigned size = dev_Value.getNbElements();

  const dim3 threads(nbThreads);
  const dim3 blocks((size + nbThreads - 1) / nbThreads);

  transformation_kernel<<<blocks,threads>>>(
      dev_Value.getDevicePointer(),
      dev_repartition.getDevicePointer(),
      dev_transformation.getDevicePointer(),
      size
  );
}
/**********************************/
/*   AL NATOUR MAZEN, M1 Info CL  */
/**********************************/