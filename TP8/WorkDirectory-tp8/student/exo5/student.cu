#include <iostream>
#include <exo5/student.h>
#include <OPP_cuda.cuh>

namespace
{
  using uchar = unsigned char;
  __global__
      void ahe_transformation_kernel(
          const float* const pixelValues,
          const unsigned* const histogramRepartition,
          float* const transformedValues,
          const unsigned totalPixels,
          const float lambda
      ) {
    const unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId < totalPixels) {
      const uchar pixelIntensity = uchar(pixelValues[threadId]); // Convert the value at index tid to an unsigned char

      // Calculate the different parts of the equation
      const float lambdaComplement = 1 / (1 + lambda);
      const float histogramValue = static_cast<float>(histogramRepartition[pixelIntensity]);
      const float lambdaRatio = lambda / (1 + lambda);
      const float sizeRatio = static_cast<float>(totalPixels) * static_cast<float>(pixelIntensity) / 256;

      // Apply the transformation
      transformedValues[threadId] = (255.f * (  (lambdaComplement * histogramValue) + (lambdaRatio * sizeRatio) ) )  / static_cast<float>(totalPixels);
    }
  }
}

void StudentWorkImpl::run_Transformation(
	const float lambda, // value to use in equation (6)
	OPP::CUDA::DeviceBuffer<float>& dev_Value,
	OPP::CUDA::DeviceBuffer<unsigned>& dev_repartition,
	OPP::CUDA::DeviceBuffer<float>& dev_transformation // or "transformed"
) {
	// SI ON MET LAMBDA SUR 0 ALORS ON REVIENT SUR L EXO D AVANT
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
/**********************************/
/*   AL NATOUR MAZEN, M1 Info CL  */
/**********************************/