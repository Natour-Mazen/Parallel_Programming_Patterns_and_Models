#include <iostream>
#include <exo1/student.h>
#include <OPP_cuda.cuh>
#include <float.h>

namespace 
{
	__device__
	float3 RGB2HSV( const uchar3 inRGB ) {
		const float R = float( inRGB.x ) / 256.f;
		const float G = float( inRGB.y ) / 256.f;
		const float B = float( inRGB.z ) / 256.f;

		const float min		= fminf( R, fminf( G, B ) );
		const float max		= fmaxf( R, fmaxf( G, B ) );
		const float delta	= max - min;

		// H
		float H;
		if		( delta < FLT_EPSILON )  
			H = 0.f;
		else if	( max == R )	
			H = 60.f * ( G - B ) / ( delta + FLT_EPSILON ) + 360.f;
		else if ( max == G )	
			H = 60.f * ( B - R ) / ( delta + FLT_EPSILON ) + 120.f;
		else					
			H = 60.f * ( R - G ) / ( delta + FLT_EPSILON ) + 240.f;
		while	( H >= 360.f )	
			H -= 360.f ;

		// S
		const float S = max < FLT_EPSILON ? 0.f : 1.f - min / max;

		// V
		const float V = max;

		return make_float3( H, S, V*256.f );
	}

	__device__
	uchar3 HSV2RGB( const float H, const float S, const float V ) {
		const float	d	= H / 60.f;
		const int	hi	= int(d) % 6;
		const float f	= d - float(hi);

		const float Vn = V / 256.f;
		const float l   = Vn * ( 1.f - S );
		const float m	= Vn * ( 1.f - f * S );
		const float n	= Vn * ( 1.f - ( 1.f - f ) * S );

		float R, G, B;

		if		( hi == 0 ) 
			{ R = Vn; G = n;	B = l; }
		else if ( hi == 1 ) 
			{ R = m; G = Vn;	B = l; }
		else if ( hi == 2 ) 
			{ R = l; G = Vn;	B = n; }
		else if ( hi == 3 ) 
			{ R = l; G = m;	B = Vn; }
		else if ( hi == 4 ) 
			{ R = n; G = l;	B = Vn; }
		else				
			{ R = Vn; G = l;	B = m; }
			
		return make_uchar3( R * 256.f, G * 256.f, B * 256.f );
	}

	__global__
	void RGB2HSV_kernel( 
		uchar3 const*const source, 
		float *const Hue,
		float *const Saturation,
		float *const Value,
		const unsigned size
	) {
		const unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
		if( tid < size )
		{
			float3 hsv( RGB2HSV(source[tid]) );
			Hue[tid] = hsv.x;
			Saturation[tid] = hsv.y;
			Value[tid] = hsv.z;
		}
	}
	
	__global__
	void HSV2RGB_kernel( 
		float const*const Hue,
		float const*const Saturation,
		float const*const Value,
		uchar3 *const result, 
		const unsigned size
	) {
		const unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
		if( tid < size )
		{
			result[tid] = HSV2RGB(Hue[tid], Saturation[tid], Value[tid]);
		}
	}
}

void StudentWorkImpl::run_RGB2HSV(
	OPP::CUDA::DeviceBuffer<uchar3>& dev_source,
	OPP::CUDA::DeviceBuffer<float>& dev_Hue,
	OPP::CUDA::DeviceBuffer<float>& dev_Saturation,
	OPP::CUDA::DeviceBuffer<float>& dev_Value,
	const unsigned width,
	const unsigned height
) {
	// TODO: map
  const unsigned size = width * height;
  const unsigned threads = 1024;
  const unsigned gridSize = (size + threads - 1) / threads;

  RGB2HSV_kernel<<<gridSize, threads>>>(dev_source.getDevicePointer(),
                                          dev_Hue.getDevicePointer(),
                                          dev_Saturation.getDevicePointer(),
                                          dev_Value.getDevicePointer(),
                                          size);
  cudaDeviceSynchronize();
}

void StudentWorkImpl::run_HSV2RGB(
	OPP::CUDA::DeviceBuffer<float>& dev_Hue,
	OPP::CUDA::DeviceBuffer<float>& dev_Saturation,
	OPP::CUDA::DeviceBuffer<float>& dev_Value,
	OPP::CUDA::DeviceBuffer<uchar3>& dev_result,
	const unsigned width,
	const unsigned height
) {
	// TODO: map
  const unsigned size = width * height;
  const unsigned threads (1024);
  const unsigned gridSize = (size + threads - 1) / threads;

  HSV2RGB_kernel<<<gridSize, threads>>>(dev_Hue.getDevicePointer(),
                                        dev_Saturation.getDevicePointer(),
                                        dev_Value.getDevicePointer(),
                                        dev_result.getDevicePointer(),
                                        size);
  cudaDeviceSynchronize();
}
