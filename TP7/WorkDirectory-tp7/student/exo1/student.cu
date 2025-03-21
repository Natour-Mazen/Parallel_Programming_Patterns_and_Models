#include <iostream>
#include <exo1/student.h>
#include <OPP_cuda.cuh>

using uchar = unsigned char;

namespace 
{
	// L'idée est de recopier le code du cours (qui est dans utils/OPP_cuda_reduce.cuh)
	
	// Mais, la différence est qu'ici la réduction se fait par bloc de 1024 pixels,
	// un peu comme une réduction par segment, mais avec des segments implicites (chaque bloc est un segment).

	// Donc, il y a uniquement des réductions par blocs de pixels en utilisant threadIdx.x.

	// Un bloc de pixel va correspondre dans ce premier exercice à un bloc de threads (1024 dans les deux cas)

	//
	__device__ 
	__forceinline__
	void loadSharedMemory(float const*const data) 
	{
		// La mémoire partagée contient des FLOAT
		float*const shared = OPP::CUDA::getSharedMemory<float>();
		// position dans le tableau
		const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
		// position dans le bloc/segment
		shared[threadIdx.x] = data[tid]; 
		__syncthreads();
	}

	//
	__device__ 
	__forceinline__
	void reduceJumpingStep(const int jump)
	{
    // Get a pointer to the shared memory for this block
    auto * const shared = OPP::CUDA::getSharedMemory<float>();
    // Get the thread ID within the block
    const auto tid = threadIdx.x;
    // If the thread ID is a multiple of twice the jump value,
    // add the element at position 'tid + jump' to the element at position 'tid'
    if((tid % (jump<<1)) == 0){
      shared[tid] += shared[tid+jump];
    }
    // Synchronize to make sure all computations at this stage are done before proceeding
    __syncthreads();
	}

	//
	__device__
	__forceinline__
	float reducePerBlock(
		float const*const source
	) {
    // Get a pointer to the shared memory for this block
    auto * const shared = OPP::CUDA::getSharedMemory<float>();
    // Load data from global memory to shared memory
    loadSharedMemory(source);
    // Perform the reduction in a loop, halving the number of active threads in each iteration
    for(int i=1; i<1024; i<<=1)
      reduceJumpingStep(i);
    // At the end of the reduction, the result is stored in the first element of the shared memory
    return shared[0];
	}

	//
	__device__
	__forceinline__
	void fillBlock(
		const float color, 
		float*const result
	) {
		const unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
		result[tid] = color;
	}

	//
	__global__
	void blockEffectKernel( 
		float const*const source, 
		float *const result
	) {
		const float sumInBlock = reducePerBlock(source);
		fillBlock(sumInBlock, result);
	}
}

// Cette fonction sera appelée trois fois pour une image donnée, car l'image est séparée en trois tableaux,
// l'un pour le rouge, l'autre pour le vert et enfin le dernier pour le bleu. 
// Cela simplifie le code et réduit la pression sur les registres ;-)
void StudentWorkImpl::run_blockEffect(
	OPP::CUDA::DeviceBuffer<float>& dev_source,
	OPP::CUDA::DeviceBuffer<float>& dev_result
) {
	const auto size = dev_source.getNbElements();
	const auto nbWarps = 32;
	dim3 threads(32*nbWarps);
	dim3 blocks(( size + threads.x-1 ) / threads.x);
	const size_t sizeSharedMemory(threads.x*sizeof(float));
	::blockEffectKernel<<<blocks, threads, sizeSharedMemory>>>(
		dev_source.getDevicePointer(),
		dev_result.getDevicePointer()
	);
}
/**********************************/
/*   AL NATOUR MAZEN, M1 Info CL  */
/**********************************/