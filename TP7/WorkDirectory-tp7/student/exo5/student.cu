#include <iostream>
#include <exo5/student.h>
#include <OPP_cuda.cuh>

using uchar = unsigned char;

namespace 
{
	// idem exo3
	template<int NB_WARPS>
	__device__ 
	__forceinline__
	void loadSharedMemoryCommutative(float const*const data) 
	{
    float *const shared = OPP::CUDA::getSharedMemory<float>();
    float sum = 0.f;
    const unsigned globalOffset = blockIdx.x * 1024;
    for (auto tid = threadIdx.x; tid < 1024; tid += 32 * NB_WARPS)
    {
      sum += data[tid + globalOffset];
    }
    const auto localThreadId = threadIdx.x;
    shared[localThreadId] = sum;
    __syncthreads();
	}

	// idem exo4
  /** This function performs a reduction step by adding elements in shared memory
   * with a certain 'jump' distance. It is executed by all threads in a block.
   * It also checks for out-of-bounds accesses.
    * */
	__device__ 
	__forceinline__
	void reduceJumpingStep(const int jump)
	{
    float *const shared = OPP::CUDA::getSharedMemory<float>();
    const auto tid = threadIdx.x;
    // If the thread ID is less than the jump value,
    // add the element at position 'tid + jump' to the element at position 'tid'
    if(tid < jump){
      shared[tid] += shared[tid+jump];
    }
    __syncthreads();
	}

	// nouvelle fonction !
  /**
   *This function performs the last reduction step within a warp.
   * It uses volatile shared memory to ensure that all memory operations are completed as expected.
   */
  template<int NB_WARPS>
	__device__ 
	__forceinline__
	void reduceLastWarp()
	{
		// attention au mot clé volatile ... essentiel !
    // Get a pointer to the shared memory for this block
    // The 'volatile' keyword is used to prevent the compiler from optimizing memory operations
		volatile float*const shared = OPP::CUDA::getSharedMemory<float>();
		const auto tid = threadIdx.x;
    // If the thread ID is less than 32 (the warp size),
    // perform the reduction within the warp
		if( tid < 32 )
		{
      // Perform the reduction in a loop, halving the number of active threads in each iteration
      // The condition checks whether there is only one warp left
      for(int i= (NB_WARPS == 1 ? 16 : 32)  ; i > 0 ; i>>=1){
        shared[tid] += shared[tid + i];
      }
		}
		__syncthreads();
	}

	
  /** This function performs a block-wise reduction on the input data
    * It uses a different number of iterations compared to the previous version
    * */
	template<int NB_WARPS>
	__device__
	__forceinline__
	float reducePerBlock(
		float const*const source
	) {
    float*const shared = OPP::CUDA::getSharedMemory<float>();
    loadSharedMemoryCommutative<NB_WARPS>(source);
    // Perform the reduction in a loop, halving the number of active threads in each iteration
    // Stop when the number of active threads is equal to the warp size
    for(int i= 32 * NB_WARPS / 2 ; i > 32; i>>=1){
      reduceJumpingStep(i);
    }
    // Perform the last reduction step within a warp
    reduceLastWarp<NB_WARPS>();
    return shared[0];
	}

	
	// idem exo3
	template<int NB_WARPS>
	__device__
	__forceinline__
	void fillBlock(
		const float color, 
		float*const result
	) {
    // calcul de l'offset du bloc : la taille est 1024
    const auto offset = blockIdx.x * 1024;
    // Get the thread ID within the block
    unsigned tid = threadIdx.x;
    // Each thread fills multiple elements in the result array with the color
    while (tid < 1024) {
      result[tid + offset] = color;
      tid += 32 * NB_WARPS;
    }
	}


	// idem exo2
	template<int NB_WARPS>
	struct EvaluateWarpNumber {
		enum { NB_BLOCKS = 1 };
	};
	template<>
	struct EvaluateWarpNumber<1> {
		enum { NB_BLOCKS = 16 };
	};
	template<>
	struct EvaluateWarpNumber<2> {
		enum { NB_BLOCKS = 8 };
	};
	template<>
	struct EvaluateWarpNumber<4> {
		enum { NB_BLOCKS = 4 };
	};
	template<>
	struct EvaluateWarpNumber<8> {
		enum { NB_BLOCKS = 4 };
	};
	template<>
	struct EvaluateWarpNumber<16> {
		enum { NB_BLOCKS = 2 };
	};
	template<int NB_WARPS=32>
	__global__
	__launch_bounds__(32*NB_WARPS , EvaluateWarpNumber<NB_WARPS>::NB_BLOCKS)
	void blockEffectKernel( 
		float const*const source, 
		float *const result
	) {
		const float sumInBlock = reducePerBlock<NB_WARPS>(source);
		fillBlock<NB_WARPS>(sumInBlock, result);
	}
}


// Attention : ici la taille des vecteurs n'est pas toujours un multiple du nombre de threads !
// Il faut donc corriger l'exemple du cours ...
void StudentWorkImpl::run_blockEffect(
	OPP::CUDA::DeviceBuffer<float>& dev_source,
	OPP::CUDA::DeviceBuffer<float>& dev_result,
	const unsigned nbWarps
) {
	const auto size = dev_source.getNbElements();
	dim3 threads(32*nbWarps);
	dim3 blocks((size + 1023) / 1024);
	const size_t sizeSharedMemory(threads.x*sizeof(float));
	switch(nbWarps) {
		case 1:
			::blockEffectKernel<1> <<<blocks, threads, sizeSharedMemory>>>(
				dev_source.getDevicePointer(),
				dev_result.getDevicePointer()
			);
			return;
		case 2:
			::blockEffectKernel<2> <<<blocks, threads, sizeSharedMemory>>>(
				dev_source.getDevicePointer(),
				dev_result.getDevicePointer()
			);
			return;
		case 4:
			::blockEffectKernel<4> <<<blocks, threads, sizeSharedMemory>>>(
				dev_source.getDevicePointer(),
				dev_result.getDevicePointer()
			);
			return;
		case 8:
			::blockEffectKernel<8> <<<blocks, threads, sizeSharedMemory>>>(
				dev_source.getDevicePointer(),
				dev_result.getDevicePointer()
			);
			return;
		case 16:
			::blockEffectKernel<16> <<<blocks, threads, sizeSharedMemory>>>(
				dev_source.getDevicePointer(),
				dev_result.getDevicePointer()
			);
			return;
		case 32:
			::blockEffectKernel<32><<<blocks, threads, sizeSharedMemory>>>(
				dev_source.getDevicePointer(),
				dev_result.getDevicePointer()
			);
			return;
		default:
			::blockEffectKernel<32><<<blocks, threads, sizeSharedMemory>>>(
				dev_source.getDevicePointer(),
				dev_result.getDevicePointer()
			);
			return;
	}
}
/**********************************/
/*   AL NATOUR MAZEN, M1 Info CL  */
/**********************************/