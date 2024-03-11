#include <iostream>
#include <exo3/student.h>
#include <OPP_cuda.cuh>

using uchar = unsigned char;

namespace
{
	// Beaucoup de solutions ici sont possibles ...
	// Celle-ci traite le bloc de pixels par morceau. Chaque pixel d'un morceau est traité par un thread.
	// On répète le processus jusqu'à avoir couvert tout le bloc de pixel.
	// NB: il faut un réduction par thread (séquentielle, avec variable privée)
	template <int NB_WARPS>
	__device__
		__forceinline__ void
		loadSharedMemoryCommutative(float const *const data)
	{
		float *const shared = OPP::CUDA::getSharedMemory<float>();
		float sum = 0.f;
		const unsigned globalOffset = blockIdx.x * 1024;
		for (auto tid = threadIdx.x; tid < 1024; tid += 32 * NB_WARPS)
		{
			// TODO
      sum += data[tid + globalOffset];
		}
		const auto localThreadId = threadIdx.x;
		shared[localThreadId] = sum;
		__syncthreads();
	}

	// idem exo2
	template <int NB_WARPS>
	__device__
		__forceinline__ void
		reduceJumpingStep(const int jump)
	{
		// TODO
    float *const shared = OPP::CUDA::getSharedMemory<float>();
    const auto tid = threadIdx.x;
    if((tid % (jump<<1)) == 0 && tid + jump < 32*NB_WARPS){
      shared[tid] += shared[tid+jump];
    }
    __syncthreads();
	}

	// Idem exo2, sauf le nom de la fonction de chargement ;-)
	template <int NB_WARPS>
	__device__
		__forceinline__ float
		reducePerBlock(
			float const *const source)
	{
		// TODO
    float*const shared = OPP::CUDA::getSharedMemory<float>();
    loadSharedMemoryCommutative<NB_WARPS>(source);
    for(int i=1; i<32*NB_WARPS; i<<=1){
      reduceJumpingStep<NB_WARPS>(i);
    }
    return shared[0];
	}

	// idem exo2
	template <int NB_WARPS>
	__device__
		__forceinline__ void
		fillBlock(
			const float color,
			float *const result)
	{
		// TODO
    // calcul de l'offset du bloc : la taille est 1024
    const auto offset = blockIdx.x * 1024;
    // TODO
    unsigned tid = threadIdx.x;

    while (tid < 1024) {
      result[tid + offset] = color;
      tid += 32 * NB_WARPS;
    }
	}

	// idem exo2
	template <int NB_WARPS>
	struct EvaluateWarpNumber
	{
		enum
		{
			NB_BLOCKS = 1
		};
	};
	template <>
	struct EvaluateWarpNumber<1>
	{
		enum
		{
			NB_BLOCKS = 16
		};
	};
	template <>
	struct EvaluateWarpNumber<2>
	{
		enum
		{
			NB_BLOCKS = 8
		};
	};
	template <>
	struct EvaluateWarpNumber<4>
	{
		enum
		{
			NB_BLOCKS = 4
		};
	};
	template <>
	struct EvaluateWarpNumber<8>
	{
		enum
		{
			NB_BLOCKS = 4
		};
	};
	template <>
	struct EvaluateWarpNumber<16>
	{
		enum
		{
			NB_BLOCKS = 2
		};
	};

	template <int NB_WARPS = 32>
	__global__
		__launch_bounds__(32 * NB_WARPS, EvaluateWarpNumber<NB_WARPS>::NB_BLOCKS) void blockEffectKernel(
			float const *const source,
			float *const result)
	{
		const float sumInBlock = reducePerBlock<NB_WARPS>(source);
		fillBlock<NB_WARPS>(sumInBlock, result);
	}
}

void StudentWorkImpl::run_blockEffect(
	OPP::CUDA::DeviceBuffer<float> &dev_source,
	OPP::CUDA::DeviceBuffer<float> &dev_result,
	const unsigned nbWarps)
{
	const auto size = dev_source.getNbElements();

	dim3 threads(32 * nbWarps);
	dim3 blocks((size + 1023) / 1024);
	const size_t sizeSharedMemory(threads.x * sizeof(float));
	switch (nbWarps)
	{
	case 1:
		::blockEffectKernel<1><<<blocks, threads, sizeSharedMemory>>>(
			dev_source.getDevicePointer(),
			dev_result.getDevicePointer());
		return;
	case 2:
		::blockEffectKernel<2><<<blocks, threads, sizeSharedMemory>>>(
			dev_source.getDevicePointer(),
			dev_result.getDevicePointer());
		return;
	case 4:
		::blockEffectKernel<4><<<blocks, threads, sizeSharedMemory>>>(
			dev_source.getDevicePointer(),
			dev_result.getDevicePointer());
		return;
	case 8:
		::blockEffectKernel<8><<<blocks, threads, sizeSharedMemory>>>(
			dev_source.getDevicePointer(),
			dev_result.getDevicePointer());
		return;
	case 16:
		::blockEffectKernel<16><<<blocks, threads, sizeSharedMemory>>>(
			dev_source.getDevicePointer(),
			dev_result.getDevicePointer());
		return;
	case 32:
		::blockEffectKernel<32><<<blocks, threads, sizeSharedMemory>>>(
			dev_source.getDevicePointer(),
			dev_result.getDevicePointer());
		return;
	default:
		::blockEffectKernel<32><<<blocks, threads, sizeSharedMemory>>>(
			dev_source.getDevicePointer(),
			dev_result.getDevicePointer());
		return;
	}
}
