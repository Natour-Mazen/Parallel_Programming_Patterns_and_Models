#pragma once

#include <StudentWork.h>
#include <vector>
#include <OPP_cuda.cuh>


class StudentWorkImpl: public StudentWork
{
public:
	StudentWorkImpl() = default; 
	StudentWorkImpl(const StudentWorkImpl&) = default;
	~StudentWorkImpl() = default;
	StudentWorkImpl& operator=(const StudentWorkImpl&) = default;

	void run_Transformation(
		const float lambda,
		OPP::CUDA::DeviceBuffer<float>& dev_Value,
		OPP::CUDA::DeviceBuffer<unsigned>& dev_repartition,
		OPP::CUDA::DeviceBuffer<float>& dev_Transformation
	);
	
};