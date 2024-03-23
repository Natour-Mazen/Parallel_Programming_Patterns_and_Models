#include <iostream>
#include <exo2/student.h>
#include <OPP_cuda.cuh>

namespace 
{
  // Functor structure for histogram computation
  struct foncteur {
    __device__
        // Operator to convert float value to unsigned integer for histogram binning
        unsigned operator()(const float& v) const {
            return static_cast<unsigned>(v);
        }
  };
}

void StudentWorkImpl::run_Histogram(
	OPP::CUDA::DeviceBuffer<float>& dev_value,
	OPP::CUDA::DeviceBuffer<unsigned>& dev_histogram,
	const unsigned width,
	const unsigned height
) {
  // Call computeHistogram function with the appropriate template parameters and functor to achieve our computeHistogram
  OPP::CUDA::computeHistogram<float, unsigned, foncteur>(dev_value, dev_histogram, foncteur());
  cudaDeviceSynchronize();
}
/**********************************/
/*   AL NATOUR MAZEN, M1 Info CL  */
/**********************************/