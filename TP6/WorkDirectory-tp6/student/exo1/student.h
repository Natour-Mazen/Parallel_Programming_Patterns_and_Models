#pragma once

#include <Functors/Sepia.h>
#include <OPP_cuda.cuh>
#include <StudentWork.h>

#include <exo1/UnaryMap.h>

class StudentWorkImpl : public StudentWork {
 public:
  StudentWorkImpl() = default;
  StudentWorkImpl(const StudentWorkImpl &) = default;
  ~StudentWorkImpl() = default;
  StudentWorkImpl &operator=(const StudentWorkImpl &) = default;

  void run_sepia(OPP::CUDA::DeviceBuffer<uchar3> &dev_inputImage,
                 OPP::CUDA::DeviceBuffer<uchar3> &dev_outputImage,
                 const unsigned imageWidth,
                 const unsigned imageHeight)
  {
    OPP::CUDA::UnaryMap(dev_inputImage.getDevicePointer(),
                        dev_outputImage.getDevicePointer(),
                        dev_inputImage.getNbElements(),
                        Functors::SepiaFunctor());
  }
};