#include <iostream>
#include <exo5/student.h>
#include <OPP_cuda.cuh>

namespace 
{
}

void StudentWorkImpl::run_Transformation(
	const float lambda, // value to use in equation (6)
	OPP::CUDA::DeviceBuffer<float>& dev_Value,
	OPP::CUDA::DeviceBuffer<unsigned>& dev_repartition,
	OPP::CUDA::DeviceBuffer<float>& dev_transformation // or "transformed"
) {
	// TODO: a map
	// NB: equation (6) is applied on the fly to the transformed value...
}
