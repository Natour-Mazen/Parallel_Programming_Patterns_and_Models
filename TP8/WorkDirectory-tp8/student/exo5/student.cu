#include <iostream>
#include <exo5/student.h>
#include <OPP_cuda.cuh>

namespace
{
  using uchar = unsigned char;
  __global__
      void ahe_transformation_kernel(
          const float* const value,
          const unsigned* const repartition,
          float* const transformation,
          const unsigned size,
          const float lambda
      ) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
      const uchar xi = uchar(value[tid] * 255.f);
      const float prob_xi = float(repartition[xi]) / float(size);
      const float prob_uniform = 1.0f / 256.0f;
      const float T_xi = (1.0f - lambda) * (prob_xi / prob_uniform) + lambda;
      transformation[tid] = T_xi;
    }
  }
}

void StudentWorkImpl::run_Transformation(
	const float lambda, // value to use in equation (6)
	OPP::CUDA::DeviceBuffer<float>& dev_Value,
	OPP::CUDA::DeviceBuffer<unsigned>& dev_repartition,
	OPP::CUDA::DeviceBuffer<float>& dev_transformation // or "transformed"
) {
	// TODO: a map
	// NB: equation (6) is applied on the fly to the transformed value...
  const unsigned nbThreads = 1024;
  const unsigned size = dev_Value.getNbElements();

  const dim3 threads(nbThreads);
  const dim3 blocks((size + nbThreads - 1) / nbThreads);

  ahe_transformation_kernel<<<blocks, threads>>>(
      dev_Value.getDevicePointer(),
      dev_repartition.getDevicePointer(),
      dev_transformation.getDevicePointer(),
      size,
      lambda
  );
}
