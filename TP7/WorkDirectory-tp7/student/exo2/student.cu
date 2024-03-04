#include <iostream>
#include <exo2/student.h>
#include <OPP_cuda.cuh>

using uchar = unsigned char;

namespace 
{
	// L'opération est associative (enfin, en toute généralité), et donc les permutations de valeurs sont interdites.
	// Seul les changements de parenthèses sont autorisées ...
	// Donc il y a deux solutions :
	// - La plus simple est d'effectuer plusieurs réductions successives par blocs
	// - La plus difficile mais efficace, et de grouper les valeurs consécutives par thread.
	// Avec cette seconde, le premier thread (0) va traiter des valeurs consécutives. Le thread suivant aussi, etc.
	// En supposant par exemple que chaque thread traite 4 valeurs, alors les 4 premiers pixels du blocs sont utilisés par
	// le thread 0, le 4 suivant par le thread 1, etc. jusqu'au thread 255 ;-)
	// NB : on suppose que le nombre de warps est une puissance de 2 (et donc divise 1024)
	template<int NB_WARPS>
	__device__ 
	__forceinline__
	void loadSharedMemoryAssociate(float const*const data) 
	{
		float*const shared = OPP::CUDA::getSharedMemory<float>();

		const auto globalOffset = 1024 * blockIdx.x;
		const auto localThreadId = threadIdx.x;
		const unsigned nbPixelsPerThread = (1024 + 32*NB_WARPS - 1) / (32*NB_WARPS);

		float sumPerThread = 0.f;

		for(unsigned i=0; i<nbPixelsPerThread; ++i) 
		{
			// indice du pixel à traiter (consécutifs pour un thread!)
			const auto pixelIdInBlock = nbPixelsPerThread * localThreadId + i;
			
			// TODO
		}
		shared[localThreadId] = sumPerThread;
		__syncthreads();
	}


	// idem exo1, sauf test de débordement
	template<int NB_WARPS>
	__device__ 
	__forceinline__
	void reduceJumpingStep(const int jump)
	{
		// TODO 
	}


	// on ne changera ici que le nombre d'itérations (10 avant, ici moins)
	template<int NB_WARPS>
	__device__
	__forceinline__
	float reducePerBlock(
		float const*const source
	) {
		// TODO
	}	
	

	// ressemble beaucoup à l'exo1 ...
	template<int NB_WARPS>
	__device__
	__forceinline__
	void fillBlock(
		const float color, 
		float*const result
	) {
		// calcul de l'offset du bloc : la taille est 1024
		const auto offset = blockIdx.x * 1024;
		// TODO
	}


	// idem exo1 with templates
	// Petite astuce pour définir le nombre de blocs en fonction du nombre de warps...
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

	// idem exo1
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


// idem exo1, sauf la taille d'un bloc de threads (en Y)
void StudentWorkImpl::run_blockEffect(
	OPP::CUDA::DeviceBuffer<float>& dev_source,
	OPP::CUDA::DeviceBuffer<float>& dev_result,
	const unsigned nbWarps
) {
	// Le nombre de warps est réduit ...
	const auto size = dev_source.getNbElements();
	// Le nombre de threads par bloc dépend du nombre de warps ;-)
	dim3 threads(32 * nbWarps); 
	// Attention : le nombre de blocs est calculer en considérant des traitements de 1024 pixels ! 
	dim3 blocks ((size + 1024-1) / 1024 );
	// le reste est classique
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