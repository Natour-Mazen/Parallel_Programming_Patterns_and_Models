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

	/**
	 * @brief Runs the Weighted Histogram Equalization from input values
	 * 
	 * This method uses two temporary device buffer to compute 
	 * the histogram of the values and the per value weight as intermediary
	 * results. 
	 * Then it apply the histogram equalization with the formula (12) from [1]. 
	 * 
	 * @param dev_inputValue input values
	 * @param dev_histo device memory for histogram of values (not set)
	 * @param dev_weight device memory for weight terms (not set)
	 * @param dev_outputValue output values after WHE
	 * @param imageWidth width of the input image (for local variance...)
	 * @param imageHeight height of the input image
	 * @param lambda penalty term of WHE
	 */
	void StudentWorkImpl::run_WHE(
		OPP::CUDA::DeviceBuffer<float>& dev_inputValue,
		OPP::CUDA::DeviceBuffer<unsigned> &dev_histo,
		OPP::CUDA::DeviceBuffer<float> &dev_weight,
		OPP::CUDA::DeviceBuffer<float>& dev_outputValue,
		const unsigned imageWidth, 
		const unsigned imageHeight, 
		const float lambda
	) ;
	
};