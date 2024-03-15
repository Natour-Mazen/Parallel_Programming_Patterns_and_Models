#include <iostream>
#include <exo2/student.h>
#include <OPP_cuda.cuh>

namespace 
{
}

void StudentWorkImpl::run_Histogram(
	OPP::CUDA::DeviceBuffer<float>& dev_value,
	OPP::CUDA::DeviceBuffer<unsigned>& dev_histogram,
	const unsigned width,
	const unsigned height
) {
	// TODO: build histogram
}
