package org.wallerlab.yoink.service;

import java.util.Random;

public class JavaMatrix implements MatrixInterface
{
	public float[] testSgemm(int n)
	{
		float alpha = 0.3f;
		float beta = 0;
		int nn = n * n;

		float h_A[] = createRandomFloatData(nn);
		float h_B[] = createRandomFloatData(nn);
			float h_C[] = createRandomFloatData(nn);
	    float h_C_ref[] = h_C.clone();

	    return sgemmJava(n, alpha, h_A, h_B, beta, h_C_ref);
	    
	}
	private  float [] sgemmJava(int n, float alpha, float A[], float B[],float beta, float C[])
	{
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				float prod = 0;
				for (int k = 0; k < n; ++k)
			    {
					prod += A[k * n + i] * B[j * n + k];
	            }
				 C[j * n + i] = alpha * prod + beta * C[j * n + i];	
	        }
        }
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