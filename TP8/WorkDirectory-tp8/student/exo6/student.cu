#include <OPP/OPP_cuda_buffer.cuh>
#include <OPP/OPP_cuda_histogram.cuh>
#include <exo6/student.h>

#include <OPP_cuda.cuh>
#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <type_traits>

#include <cuda_runtime.h>
#include <cmath>
#include <OPP/OPP_cuda_buffer.cuh>
#include <OPP/OPP_cuda_histogram.cuh>
#include <exo6/student.h>

#include <OPP_cuda.cuh>
#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <type_traits>

namespace {
  using uchar = unsigned char;

  __global__
      void buildHistogramAndVarianceSum_kernel(
          const float* const dev_inputValue,
          unsigned* const dev_histo,
          float* const dev_weight,
          const unsigned size,
          const unsigned imageWidth) {

    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
      const uchar xi = uchar(dev_inputValue[tid]);
      atomicAdd(&dev_histo[xi], 1u);

      const unsigned row = tid / imageWidth;
      const unsigned col = tid % imageWidth;
      const unsigned left_idx = row * imageWidth + ((col + imageWidth - 1) % imageWidth);
      const unsigned right_idx = row * imageWidth + ((col + 1) % imageWidth);

      const float left_value = dev_inputValue[left_idx];
      const float right_value = dev_inputValue[right_idx];
      const float local_variance = (left_value - dev_inputValue[tid]) * (left_value - dev_inputValue[tid]) +
                                   (right_value - dev_inputValue[tid]) * (right_value - dev_inputValue[tid]);
      atomicAdd(&dev_weight[xi], local_variance / 2.0f);
    }
  }

  __global__
      void buildCumulativeDistributionFunction_kernel(
          unsigned* const dev_cdf,
          const float* const dev_weight,
          const float lambda,
          const unsigned size) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
      const float weighted_count = ( (float(dev_cdf[tid]) + (1.0f - lambda)) / ((lambda + dev_weight[tid])) );
      dev_cdf[tid] = static_cast<unsigned>(weighted_count);
    }
  }

  __global__
      void applyTransformation_kernel(
          const float* const dev_inputValue,
          const unsigned* const dev_cdf,
          float* const dev_outputValue,
          const unsigned size) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
      const uchar xi = uchar(dev_inputValue[tid]);
      const float cdf_sum = float(dev_cdf[255]);
      dev_outputValue[tid] = float(dev_cdf[xi]) / cdf_sum;
    }
  }

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
    const unsigned nbThreads = 1024;
    const unsigned size = dev_inputValue.getNbElements();

    const dim3 threads(nbThreads);
    const dim3 blocks((size + nbThreads - 1) / nbThreads);

    buildHistogramAndVarianceSum_kernel<<<blocks, threads>>>(
        dev_inputValue.getDevicePointer(),
        dev_cdf.getDevicePointer(),
        dev_weight.getDevicePointer(),
        size,
        imageWidth);
    cudaDeviceSynchronize();
  }

  void buildCumulativeDistributionFunction(OPP::CUDA::DeviceBuffer<unsigned> &dev_cdf,
                                           OPP::CUDA::DeviceBuffer<float> &dev_weight,
                                           const float lambda,
                                           const unsigned size)
  {
    // TODO
    const unsigned nbThreads = 1024;

    const dim3 threads(nbThreads);
    const dim3 blocks((size + nbThreads - 1) / nbThreads);

    buildCumulativeDistributionFunction_kernel<<<blocks, threads>>>(
        dev_cdf.getDevicePointer(),
        dev_weight.getDevicePointer(),
        lambda,
        size);
    cudaDeviceSynchronize();
  }

  void applyTransformation(OPP::CUDA::DeviceBuffer<float> &dev_inputValue,
                           OPP::CUDA::DeviceBuffer<unsigned> &dev_cdf,
                           OPP::CUDA::DeviceBuffer<float> &dev_outputValue)
  {
    // TODO
    const unsigned nbThreads = 1024;
    const unsigned size = dev_inputValue.getNbElements();

    const dim3 threads(nbThreads);
    const dim3 blocks((size + nbThreads - 1) / nbThreads);

    applyTransformation_kernel<<<blocks, threads>>>(
        dev_inputValue.getDevicePointer(),
        dev_cdf.getDevicePointer(),
        dev_outputValue.getDevicePointer(),
        size);
    cudaDeviceSynchronize();
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

//
//namespace {
//  using uchar = unsigned char;
//
//  __global__
//      void buildHistogramAndVarianceSum_kernel(
//          const float* const dev_inputValue,
//          unsigned* const dev_histo,
//          float* const dev_weight,
//          const unsigned size,
//          const unsigned imageWidth) {
//
//    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
//
//    if (tid < size) {
//      const uchar xi = uchar(dev_inputValue[tid] * 255.0F);
//      atomicAdd(&dev_histo[xi], 1u);
//
//      const unsigned row = tid / imageWidth;
//      const unsigned col = tid % imageWidth;
//      const unsigned left_idx = row * imageWidth + ((col + imageWidth - 1) % imageWidth);
//      const unsigned right_idx = row * imageWidth + ((col + 1) % imageWidth);
//
//      const float left_value = dev_inputValue[left_idx];
//      const float right_value = dev_inputValue[right_idx];
//      const float local_variance = 0.5 * ((left_value - dev_inputValue[tid]) * (left_value - dev_inputValue[tid]) +
//                                   (right_value - dev_inputValue[tid]) * (right_value - dev_inputValue[tid]));
//
//      atomicAdd(&dev_weight[xi], local_variance );
//    }
//
//  }
//
//  __global__
//      void buildCumulativeDistributionFunction_kernel(
//          unsigned* const dev_cdf,
//          const float* const dev_weight,
//          const float lambda,
//          const unsigned size) {
//       const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
//       if (tid < size) {
//         const float weighted_count = ( (float(dev_cdf[tid]) * (1.0f - lambda))- (lambda * dev_weight[tid]) );
//         dev_cdf[tid] = static_cast<unsigned>(weighted_count);
//        }
//  }
//
//  __global__
//      void applyTransformation_kernel(
//          const float* const dev_inputValue,
//          const unsigned* const dev_cdf,
//          float* const dev_outputValue,
//          const unsigned size) {
//    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
//
//    if (tid < size) {
////      const uchar xi = uchar(dev_inputValue[tid]);
////      const float cdf_sum = float(dev_cdf[255]);
////      dev_outputValue[tid] = float(dev_cdf[xi]) / cdf_sum;
//      const uchar xi = uchar(dev_inputValue[tid] * 255.0f); // Multiply by 255 before casting
//      float cdf_max = 0.0f;
//      for (int i = 0; i < 256; ++i) { // Find the maximum value in the CDF
//        cdf_max = max(cdf_max, float(dev_cdf[i]));
//      }
//      dev_outputValue[tid] = float(dev_cdf[xi]) / cdf_max; // Normalize by the maximum value in the CDF
//    }
//  }
//
//  /**
//   * @brief print a device buffer on standard output
//   *
//   * @tparam T Datatype of the data stored into the DeviceDuffer
//   * @param msg Message to display first (before the data)
//   * @param deviceBuffer Contains the data to display
//   */
//  template <typename T>
//  void print(std::string &msg, OPP::CUDA::DeviceBuffer<T> &deviceBuffer)
//  {
//    const auto size = deviceBuffer.getNbElements();
//    std::vector<T> hostVector(size);
//    deviceBuffer.copyToHost(hostVector.data());
//    std::cout << "======= " << msg << " of size " << size << " =====\n";
//    for (unsigned i{0u}; i < size; ++i) {
//      std::cout << hostVector[i] << " ";
//      if ((i % 16u) == 15u) {
//        std::cout << "\n";
//      }
//    }
//  }
//
//  void buildHistogramAndVarianceSum(OPP::CUDA::DeviceBuffer<float> &dev_inputValue,
//                                    OPP::CUDA::DeviceBuffer<unsigned> &dev_cdf,
//                                    OPP::CUDA::DeviceBuffer<float> &dev_weight,
//                                    const unsigned imageWidth)
//  {
//    const unsigned nbThreads = 1024;
//    const unsigned size = dev_inputValue.getNbElements();
//
//    const dim3 threads(nbThreads);
//    const dim3 blocks((size + nbThreads - 1) / nbThreads);
//
//    buildHistogramAndVarianceSum_kernel<<<blocks, threads>>>(
//        dev_inputValue.getDevicePointer(),
//        dev_cdf.getDevicePointer(),
//        dev_weight.getDevicePointer(),
//        size,
//        imageWidth);
//
//    cudaDeviceSynchronize();
//  }
//
//  void buildCumulativeDistributionFunction(OPP::CUDA::DeviceBuffer<unsigned> &dev_cdf,
//                                           OPP::CUDA::DeviceBuffer<float> &dev_weight,
//                                           const float lambda,
//                                           const unsigned size)
//  {
//    const unsigned nbThreads = 1024;
//
//    const dim3 threads(nbThreads);
//    const dim3 blocks((size + nbThreads - 1) / nbThreads);
//
//    buildCumulativeDistributionFunction_kernel<<<blocks, threads>>>(
//        dev_cdf.getDevicePointer(),
//        dev_weight.getDevicePointer(),
//        lambda,
//        size);
//    cudaDeviceSynchronize();
//  }
//
//  void applyTransformation(OPP::CUDA::DeviceBuffer<float> &dev_inputValue,
//                           OPP::CUDA::DeviceBuffer<unsigned> &dev_cdf,
//                           OPP::CUDA::DeviceBuffer<float> &dev_outputValue)
//  {
//    const unsigned nbThreads = 1024;
//    const unsigned size = dev_inputValue.getNbElements();
//
//    const dim3 threads(nbThreads);
//    const dim3 blocks((size + nbThreads - 1) / nbThreads);
//
//    applyTransformation_kernel<<<blocks, threads>>>(
//        dev_inputValue.getDevicePointer(),
//        dev_cdf.getDevicePointer(),
//        dev_outputValue.getDevicePointer(),
//        size);
//    cudaDeviceSynchronize();
//  }
//} // namespace
//
//void StudentWorkImpl::run_WHE([[maybe_unused]] OPP::CUDA::DeviceBuffer<float> &dev_inputValue,
//                              [[maybe_unused]] OPP::CUDA::DeviceBuffer<unsigned> &dev_histo,
//                              [[maybe_unused]] OPP::CUDA::DeviceBuffer<float> &dev_weight,
//                              [[maybe_unused]] OPP::CUDA::DeviceBuffer<float> &dev_outputValue,
//                              const unsigned imageWidth,
//                              const unsigned imageHeight,
//                              const float lambda)
//{
//  // 1. calcul par valeur dans [0..255/256] de l'histogramme ET de la somme des variances/valeur
//  ::buildHistogramAndVarianceSum(dev_inputValue, dev_histo, dev_weight, imageWidth);
//
//   //::print<unsigned>(std::string("histo"), dev_histo); // for debug, if needed
//   //::print(std::string("weight"), dev_weight); // for debug, if needed
//
//  // 2. calcul de la CDF (dans histo pour économiser de la mémoire)
//  ::buildCumulativeDistributionFunction(dev_histo, dev_weight, lambda, imageWidth * imageHeight);
//
//  // 3. application de la transformation...
//  ::applyTransformation(dev_inputValue, dev_histo, dev_outputValue);
//}
