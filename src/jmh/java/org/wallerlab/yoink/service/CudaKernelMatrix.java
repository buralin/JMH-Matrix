package org.wallerlab.yoink.service;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

import java.util.Random;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;

public class CudaKernelMatrix implements MatrixInterface
{
	public float[] testSgemm(int n) 
	{
			float alpha = 0.3f;
			float beta = 0;
			int nn = n * n;

			float h_A[] = createRandomFloatData(nn);
			float h_B[] = createRandomFloatData(nn);

		    return cudaCalculationKernel(n, alpha, h_A, h_B, beta);
	}
	public float [] cudaCalculationKernel (int n, float alpha, float A[], float B[],float beta)
	{ 
			// Enable exceptions and omit all subsequent error checks
	       JCudaDriver.setExceptionsEnabled(true);
	        alpha = 1;
	        beta = 1;
	        
	        int nn = n*n;
	        
	       // Initialize the driver and create a context for the first device.
	       cuInit(0);
	       CUdevice device = new CUdevice();
	       cuDeviceGet(device, 0);
	       CUcontext context = new CUcontext();
	       cuCtxCreate(context, 0, device);
	       
	       
	       // Load the ptx file.
	       CUmodule module = new CUmodule();
	       cuModuleLoad(module, "MatrixMulKernel.ptx");
	       // Obtain a function pointer to the "add" function.
	       CUfunction function = new CUfunction();
	       cuModuleGetFunction(function, module, "matrixMulkernel");
	       
	       
	       // Allocate the device input data, and copy the
	       // host input data to the device
	       CUdeviceptr d_A = new CUdeviceptr();
	       cuMemAlloc(d_A, nn * Sizeof.FLOAT);
	       cuMemcpyHtoD(d_A, Pointer.to(A),nn * Sizeof.FLOAT);
	       
	       CUdeviceptr d_B = new CUdeviceptr();
	       cuMemAlloc(d_B, nn  * Sizeof.FLOAT);
	       cuMemcpyHtoD(d_B, Pointer.to(B),nn * Sizeof.FLOAT);
	       
	       CUdeviceptr d_C = new CUdeviceptr();
	       cuMemAlloc(d_C, nn * Sizeof.FLOAT);
	       
	       // Set up the kernel parameters: A pointer to an array
	       // of pointers which point to the actual values.
	       Pointer kernelParameters = Pointer.to(Pointer.to(d_A),Pointer.to(d_B),Pointer.to(d_C));
	       
	       // Call the kernel function.
	       int blockSizeX = 32;
	       int blockSizeY = 32;
	       int gridSizeX = (nn/32)+1;
	       int gridSizeY = 1;
	       cuLaunchKernel(function,
	           gridSizeX,  gridSizeY , 1,               // Grid dimension
	           blockSizeX, blockSizeY, 1,      // Block dimension
	           0, null,                        // Shared memory size and stream
	           kernelParameters, null          // Kernel- and extra parameters
	       );
	       
	       
	       // Allocate host output memory and copy the device output
	       // to the host.
	       float C [] = new float [nn];
	       cuMemcpyDtoH(Pointer.to(C), d_C, nn * Sizeof.FLOAT);
	       
	       cuMemFree(d_A);
	       cuMemFree(d_B);
	       cuMemFree(d_C);
	       return C;
			
	}
	private static float[] createRandomFloatData(int n)
	{
		Random random = new Random();
		float x[] = new float[n];
		for (int i = 0; i < n; i++)
		{
			x[i] = random.nextFloat();
		}
		return x;
	}
}
