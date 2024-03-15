#include <OPP/OPP_cuda_buffer.cuh>
#include <OPP/OPP_cuda_histogram.cuh>
#include <exo6/student.h>

#include <OPP_cuda.cuh>
#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <type_traits>

namespace {

  /**
   * @brief print a device buffer on standard output
   * 
   * @tparam T Datatype of the data stored into the DeviceDuffer
   * @param msg Message to display first (before the data)
   * @param deviceBuffer Contains the data to display
   */
  template <typename T>
  void print(std::string &msg, OPP::CUDA::DeviceBuffer<T> &deviceBuffer)
  {
    const auto size = deviceBuffer.getNbElements();
    std::vector<T> hostVector(size);
    deviceBuffer.copyToHost(hostVector.data());
    std::cout << "======= " << msg << " of size " << size << " =====\n";
    for (unsigned i{0u}; i < size; ++i) {
      std::cout << hostVector[i] << " ";
      if ((i % 16u) == 15u) {
        std::cout << "\n";
      }
    }
  }

  void buildHistogramAndVarianceSum(OPP::CUDA::DeviceBuffer<float> &dev_inputValue,
                                    OPP::CUDA::DeviceBuffer<unsigned> &dev_cdf,
                                    OPP::CUDA::DeviceBuffer<float> &dev_weight,
                                    const unsigned imageWidth)
  {
    // TODO
  }

  void buildCumulativeDistributionFunction(OPP::CUDA::DeviceBuffer<unsigned> &dev_cdf,
                                           OPP::CUDA::DeviceBuffer<float> &dev_weight,
                                           const float lambda,
                                           const unsigned size)
  {
    // TODO
  }

  void applyTransformation(OPP::CUDA::DeviceBuffer<float> &dev_inputValue,
                           OPP::CUDA::DeviceBuffer<unsigned> &dev_cdf,
                           OPP::CUDA::DeviceBuffer<float> &dev_outputValue)
  {
    // TODO
  }
} // namespace

void StudentWorkImpl::run_WHE([[maybe_unused]] OPP::CUDA::DeviceBuffer<float> &dev_inputValue,
                              [[maybe_unused]] OPP::CUDA::DeviceBuffer<unsigned> &dev_histo,
                              [[maybe_unused]] OPP::CUDA::DeviceBuffer<float> &dev_weight,
                              [[maybe_unused]] OPP::CUDA::DeviceBuffer<float> &dev_outputValue,
                              const unsigned imageWidth,
                              const unsigned imageHeight,
                              const float lambda)
{
  // 1. calcul par valeur dans [0..255/256] de l'histogramme ET de la somme des variances/valeur
  ::buildHistogramAndVarianceSum(dev_inputValue, dev_histo, dev_weight, imageWidth);

  // ::print(std::string("histo"), dev_histo); // for debug, if needed
  // ::print(std::string("weight"), dev_weight); // for debug, if needed

  // 2. calcul de la CDF (dans histo pour économiser de la mémoire)
  ::buildCumulativeDistributionFunction(dev_histo, dev_weight, lambda, imageWidth * imageHeight);

  // 3. application de la transformation...
  ::applyTransformation(dev_inputValue, dev_histo, dev_outputValue);
}
