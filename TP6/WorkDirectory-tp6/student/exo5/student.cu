#include <iostream>
#include <exo5/student.h>
#include <OPP_cuda.cuh>

// Vous utiliserez ici les types uchar3 et float3 (internet : CUDA uchar3)
namespace 
{
	// Addition de deux "float3"
	[[maybe_unused]] 
	__device__ 
	float3 operator+(const float3 &a, const float3 &b) 
	{
		return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
	}

  __device__
      float3 operator*(const float &a, const uchar3 &b)
  {
    // Returns a new float3 where each component is the product of the corresponding component of b and a
    return make_float3(a* b.x, a * b.y, a * b.z);
  }
  __global__
      void kernelFilter(
          const uchar3* const dev_input,
          uchar3* const dev_output,
          float* filter,
          const unsigned imageWidth,
          const unsigned imageHeight,
          const unsigned filterWidth,
          size_t size
      ) {
    // Calculate the global thread ID for x and y dimensions
    const unsigned tidX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned tidY = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread ID is within the size of the image
    if (tidX < imageWidth && tidY < imageHeight) {
      const unsigned tid = tidX + tidY * imageWidth;
      float3 res = make_float3(0.0f, 0.0f, 0.0f);
      // Apply the filter to each pixel in the neighborhood
      for (unsigned i = 0; i < filterWidth; i++) {
        int X = tidX + i - filterWidth / 2;
        // Handle edge cases by mirroring the pixel position if it's outside the image boundaries
        if (X < 0) {
          X = -X;
        } else if (X > imageWidth) {
          X = imageWidth - (X - imageWidth);
        }

        for (unsigned j = 0; j < filterWidth; j++) {
          int Y = tidY + j - filterWidth / 2;
          // Handle edge cases by mirroring the pixel position if it's outside the image boundaries
          if (Y < 0) {
            Y = -Y;
          } else if (Y > imageHeight) {
            Y = imageHeight - (Y - imageHeight);
          }
          // Apply the filter to the pixel and add the result to the accumulated result
          res = res + filter[i * filterWidth + j] * dev_input[X + Y * imageWidth];
        }
      }
      // Write the filtered pixel to the output image
      dev_output[tid] = make_uchar3(
          static_cast<unsigned char>(res.x),
          static_cast<unsigned char>(res.y),
          static_cast<unsigned char>(res.z));
    }
  }
}

void StudentWorkImpl::run_filter(
	OPP::CUDA::DeviceBuffer<uchar3>& dev_inputImage,
	OPP::CUDA::DeviceBuffer<uchar3>& dev_outputImage,
	OPP::CUDA::DeviceBuffer<float>& dev_filter,
	const unsigned imageWidth, 
	const unsigned imageHeight,
	const unsigned filterWidth
) {
  // Get the total number of pixels in the image
  const unsigned size = dev_inputImage.getNbElements();
  // Define the number of threads per block
  const dim3 threads(32, 32);
  // Calculate the number of blocks needed to cover all pixels
  const dim3 blocs((imageWidth + 32 - 1) / 32,
                   (imageHeight + 32 - 1) / 32);
  // Launch the filter kernel
  kernelFilter<<<blocs, threads>>>(
      dev_inputImage.getDevicePointer(),
      dev_outputImage.getDevicePointer(),
      dev_filter.getDevicePointer(),
      imageWidth,
      imageHeight,
      filterWidth,
      size
  );
  // Synchronize the device to ensure all threads have finished before proceeding
  cudaDeviceSynchronize();
}
/**********************************/
/*   AL NATOUR MAZEN, M1 Info CL  */
/**********************************/
