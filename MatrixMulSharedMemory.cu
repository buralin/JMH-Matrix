
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

extern "C"


__global__ void  matrixMulkernelShared(float * A, float *B, int wA, int wB, float *C)
{
	//BLOCK index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	//Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	//Index of the first sub-matrix of A processed by the block
	int aBegin = wA * 16 * by;
	//Index of the last sub-matrix of A processed by the block
	int aEnd = aBegin + wA - 1;
	//Step size used to iterate thorugh the sub-matrices of A
	int aStep = 16;
	//Index of the first sub-matrix of B processed by the block
	int bBegin = 16 * bx;
	//Step size used to iterate through the sub-matrices of B
	int bStep = 16 * wB;
	// The element of the block sub-matrix that is computed 
	// by the thread 
	float Csub = 0;
	// Loop over all the sub-matrices of A and B required to 
	// compute the block sub-matrix 
	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
	{
		// Shared memory for the sub-matrix of A 
		__shared__ float As[16][16];
		// Shared memory for the sub-matrix of B 
		__shared__ float Bs[16][16];
		// Load the matrices from global memory to 	shared memory;
		// each thread loads one element of each ma	trix
		As[ty][tx] = A[a + wA * ty + tx];
		Bs[ty][tx] = B[b + wB * ty + tx];
		// Synchronize to make sure the matrices are loaded
		__syncthreads();
		// Multiply the two matrices together; 
		// each thread computes one element 
		// of the block sub-matrix 
		for (int k = 0; k < 16; ++k)
		{
			Csub += As[ty][k] * Bs[k][tx];
			// Synchronize to make sure that the preceding
			// computation is done before loading two new
			// sub-matrices of A and B in the next iteration
				__syncthreads();
		}

	}
	// Write the block sub-matrix to global memory;
	// each thread writes one element 
	int	c = wB * 16 * by + 16 * bx;
	C[c + wB * ty + tx] = Csub;

}
