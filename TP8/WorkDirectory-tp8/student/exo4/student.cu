#include <iostream>
#include <exo4/student.h>
#include <OPP_cuda.cuh>

namespace 
{
}

void StudentWorkImpl::run_Transformation(
	OPP::CUDA::DeviceBuffer<float>& dev_Value,
	OPP::CUDA::DeviceBuffer<unsigned>& dev_repartition,
	OPP::CUDA::DeviceBuffer<float>& dev_transformation // or "transformed"
) {
	// TODO
}
