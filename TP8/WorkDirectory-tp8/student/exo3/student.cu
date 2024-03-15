#include <iostream>
#include <exo3/student.h>
#include <OPP_cuda.cuh>

namespace 
{
}

void StudentWorkImpl::run_Repartition(
	OPP::CUDA::DeviceBuffer<unsigned>& dev_histogram,
	OPP::CUDA::DeviceBuffer<unsigned>& dev_repartition
) {
	// TODO: build CDF (inclusive scan!!)
}
