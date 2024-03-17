#include <iostream>
#include <exo4/student.h>
#include <OPP_cuda.cuh>

namespace 
{
  using uchar = unsigned char;

  __global__
      void transformation_kernel(
          const float* const value,
          const unsigned* const repartition,
          float* const transformation,
          const unsigned size
      ){
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  /* if (tid < size) {
      const uchar xi = uchar(value[tid] * 256.f);
      const float prob_xi = float(repartition[xi]) / float(size);
      const float prob_uniform = 1.0f / 256.0f;
      const float T_xi = (prob_xi / prob_uniform) * (255.0f / (prob_xi / prob_uniform + 1.0f));
      transformation[tid] = T_xi;
    }*/

    if (tid < size) {
      const uchar xi = uchar(value[tid] * 256.f);
      transformation[tid] = (255.f * float(repartition[xi])) / (256.f * size);
    }
  }
}

void StudentWorkImpl::run_Transformation(
	OPP::CUDA::DeviceBuffer<float>& dev_Value,
	OPP::CUDA::DeviceBuffer<unsigned>& dev_repartition,
	OPP::CUDA::DeviceBuffer<float>& dev_transformation // or "transformed"
) {
	// TODO
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
