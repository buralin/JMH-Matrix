package org.wallerlab.yoink;


import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.wallerlab.yoink.service.CublasMatrix;
import org.wallerlab.yoink.service.CudaKernelMatrix;
import org.wallerlab.yoink.service.CudaKernelSharedMatrix;
import org.wallerlab.yoink.service.JavaMatrix;
import org.wallerlab.yoink.service.MatrixInterface;



public class MethodStarter {
	


	public float[] MatrixCalculationJava (int n)
	{
		MatrixInterface dc = new JavaMatrix();
		return dc.testSgemm(n);
	}
	public float [] MatrixCalculationCublas (int n)
	{
		MatrixInterface dc = new CublasMatrix();
		return dc.testSgemm(n);
	}
	public float [] MatrixCalculationCudaKernel (int n)
	{
		MatrixInterface dc = new CudaKernelMatrix();
		return dc.testSgemm(n);
	}
	public float [] MatrixCalculationCudaKernelShared (int n)
	{
		MatrixInterface dc = new CudaKernelSharedMatrix();
		return dc.testSgemm(n);
	}
}
