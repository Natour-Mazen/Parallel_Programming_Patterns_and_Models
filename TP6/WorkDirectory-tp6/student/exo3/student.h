#ifndef EXO3_STUDENT_H
#define EXO3_STUDENT_H

#include <Functors/MapGatherFunctor.h>
#include <OPP_cuda.cuh>
#include <StudentWork.h>

#include <exo3/Gather.h>

class StudentWorkImpl : public StudentWork {
 public:
  StudentWorkImpl() = default;
  StudentWorkImpl(const StudentWorkImpl &) = default;
  ~StudentWorkImpl() = default;
  StudentWorkImpl &operator=(const StudentWorkImpl &) = default;

  void run_thumbnail_gather(OPP::CUDA::DeviceBuffer<uchar3> &dev_inputImage,
                            OPP::CUDA::DeviceBuffer<uchar3> &dev_outputImage,
                            OPP::CUDA::DeviceBuffer<uchar2> &dev_map,
                            const unsigned imageWidth,
                            const unsigned imageHeight)
  {
    Functors::MapGatherFunctor<3> map(dev_map.getDevicePointer(), imageWidth, imageHeight);

    OPP::CUDA::Gather(dev_inputImage, dev_outputImage, map);
  }
};

#endif