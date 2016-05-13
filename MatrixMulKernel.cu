
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

extern "C"

__global__ void matrixMulkernel(float * A, float *B, float *C)
{
	int I = blockIdx.x * blockDim.x + threadIdx.x;
	int J = blockIdx.y*blockDim.y + threadIdx.y;
	int N = blockDim.y*gridDim.y;

	if ((I < N) && (J < N))
	{
		float _c = 0;
		for (unsigned int k = 0; k < N; k++)
		{
			float a = A[I*N + k];
			float b = B[k*N + J];
			_c += a*b;
		}
	
		C[I*N + J] = _c;
	}


}