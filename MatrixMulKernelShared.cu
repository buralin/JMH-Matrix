
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

extern "C"


__global__ void  matrixMulkernelShared(float * A, float *B, float *C, int N)
{
	const unsigned int bx = blockIdx.x;
	const unsigned int by = blockIdx.y;
	const unsigned int tx = threadIdx.x;
	const unsigned int ty = threadIdx.y;
	const unsigned int I = 16 * bx + tx;
	const unsigned int J = 16 * by + ty;

	__shared__ float a[16][16];
	__shared__ float b[16][16];

	float sum = 0;

	for (int m; m < N / 16; m++)
	{
		a[ty][tx] = A[J*N + (m * 16 + tx)];
		b[ty][tx] = A[I + (m * 16 + ty)*N];

		__syncthreads();

		for (int k = 0; k < 16; k++)
		{
			sum += a[ty][k] * b[k][ty];
			__syncthreads();
		}
		C[J*N + I] = sum;
	}
}