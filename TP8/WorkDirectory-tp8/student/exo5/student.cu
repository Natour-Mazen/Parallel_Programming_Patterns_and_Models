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
      )
  {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size)
    {
      const float v = value[tid];
      const float u = static_cast<float>(tid) / static_cast<float>(size);
      const float h = static_cast<float>(repartition[tid]) / static_cast<float>(size);

      transformation[tid] = (1.0f - lambda) * v + lambda * ((v - h) / (1.0f - h) + u);
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

  // Mappage des valeurs des pixels en float
  OPP::CUDA::DeviceBuffer<float> dev_ValueMapped(size);
 // OPP::CUDA::inclusiveScan<float,>(dev_Value, dev_ValueMapped, [](uchar v) { return static_cast<float>(v) / 255.0f; });

  OPP::CUDA::DeviceBuffer<float> dev_TransformationMapped(size);

  const dim3 threads(nbThreads);
  const dim3 blocks((size + nbThreads - 1) / nbThreads);

  ahe_transformation_kernel<<<blocks, threads>>>(
      dev_Value.getDevicePointer(),
      dev_repartition.getDevicePointer(),
      dev_TransformationMapped.getDevicePointer(),
      size,
      lambda
  );
}
