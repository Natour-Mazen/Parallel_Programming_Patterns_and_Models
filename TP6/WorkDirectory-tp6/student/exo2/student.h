#pragma once

#include <Functors/Thumbnail.h>
#include <Iterators/OnBorder.h>
#include <OPP_cuda.cuh>
#include <StudentWork.h>
#include <cstdint>

#include <exo2/BinaryMap.h>

class StudentWorkImpl : public StudentWork {
 public:
  StudentWorkImpl() = default;
  StudentWorkImpl(const StudentWorkImpl &) = default;
  ~StudentWorkImpl() = default;
  StudentWorkImpl &operator=(const StudentWorkImpl &) = default;

  void run_thumbnail(OPP::CUDA::DeviceBuffer<uchar3> &dev_inputImage,
                     OPP::CUDA::DeviceBuffer<uchar3> &dev_outputImage,
                     const uchar3 borderColor,
                     const std::uint32_t borderSize,
                     const std::uint32_t imageWidth,
                     const std::uint32_t imageHeight)
  {
    OPP::CUDA::BinaryMap(dev_inputImage.getDevicePointer(),
                         Iterators::OnBorderIterator<3u>{borderSize, imageWidth, imageHeight},
                         dev_outputImage.getDevicePointer(),
                         dev_inputImage.getNbElements(),
                         Functors::FunctorThumbnail{borderColor});
  }
};