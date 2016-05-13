package org.wallerlab.yoink.service;

import java.util.Random;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;

public class CublasMatrix implements MatrixInterface
{
		 public float [] testSgemm(int n)
		    {
		        float alpha = 0.3f;
		        float beta = 0;
		        int nn = n * n;
		        //System.out.println("Creating input data...");
		        float h_A[] = createRandomFloatData(nn);
		        float h_B[] = createRandomFloatData(nn);
		        float h_C[] = createRandomFloatData(nn);
		        float h_C_ref[] = h_C.clone();
		        
		        return sgemmJCublas(n, alpha, h_A, h_B, beta, h_C);
		        
		        
		    }

		    public  float [] sgemmJCublas(int n, float alpha, float A[], float B[],
		                    float beta, float C[])
		    {
		        int nn = n * n;

		        // Initialize JCublas
		        JCublas.cublasInit();

		        // Allocate memory on the device
		        Pointer d_A = new Pointer();
		        Pointer d_B = new Pointer();
		        Pointer d_C = new Pointer();
		        JCublas.cublasAlloc(nn, Sizeof.FLOAT, d_A);
		        JCublas.cublasAlloc(nn, Sizeof.FLOAT, d_B);
		        JCublas.cublasAlloc(nn, Sizeof.FLOAT, d_C);

		        // Copy the memory from the host to the device
		        JCublas.cublasSetVector(nn, Sizeof.FLOAT, Pointer.to(A), 1, d_A, 1);
		        JCublas.cublasSetVector(nn, Sizeof.FLOAT, Pointer.to(B), 1, d_B, 1);
		        JCublas.cublasSetVector(nn, Sizeof.FLOAT, Pointer.to(C), 1, d_C, 1);

		        // Execute sgemm
		        JCublas.cublasSgemm(
		            'n', 'n', n, n, n, alpha, d_A, n, d_B, n, beta, d_C, n);

		        // Copy the result from the device to the host
		        JCublas.cublasGetVector(nn, Sizeof.FLOAT, d_C, 1, Pointer.to(C), 1);
		        

		        // Clean up
		        JCublas.cublasFree(d_A);
		        JCublas.cublasFree(d_B);
		        JCublas.cublasFree(d_C);
		        return C;
		    }
		    public float[] createRandomFloatData(int n)
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