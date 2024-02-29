#pragma once

#include <Functors/MapScatterFunctor.h>
#include <OPP_cuda.cuh>
#include <StudentWork.h>

#include <exo4/Scatter.h>


class StudentWorkImpl : public StudentWork {
 public:
  StudentWorkImpl() = default;
  StudentWorkImpl(const StudentWorkImpl &) = default;
  ~StudentWorkImpl() = default;
  StudentWorkImpl &operator=(const StudentWorkImpl &) = default;

  void run_thumbnail_scatter(OPP::CUDA::DeviceBuffer<uchar3> &dev_inputImage,
                             OPP::CUDA::DeviceBuffer<uchar3> &dev_outputImage,
                             OPP::CUDA::DeviceBuffer<uchar2> &dev_map,
                             const unsigned imageWidth,
                             const unsigned imageHeight)
  {
    Functors::MapScatterFunctor<3> map(dev_map.getDevicePointer(), imageWidth, imageHeight);

    OPP::CUDA::Scatter(dev_inputImage, dev_outputImage, map);
  }
};