package org.wallerlab.yoink;


import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;


@State(Scope.Thread)
public class SetOfBenchmarks {
	MethodStarter starter = new MethodStarter();
	int n;
	@Setup
	public void setup() 
	{
	  n = 1100;
	  System.out.println("MATRIX SIZE N: " + n);
	}
	
	@Benchmark
    @Warmup(iterations = 1, time = 1)
    @Measurement(iterations = 1, time = 3)
    @BenchmarkMode(Mode.SingleShotTime)
	public void  MatrixCublas (Blackhole bh){
		bh.consume(starter.MatrixCalculationCublas(n));
	}
	@Benchmark
    @Warmup(iterations = 1, time = 1)
    @Measurement(iterations = 1, time = 3)
    @BenchmarkMode(Mode.SingleShotTime)
	public void  MatrixJava (Blackhole bh){
		bh.consume(starter.MatrixCalculationJava(n));
	}
	@Benchmark
    @Warmup(iterations = 1, time = 1)
    @Measurement(iterations = 1, time = 3)
    @BenchmarkMode(Mode.SingleShotTime)
	public void  MatrixCudaKernel (Blackhole bh){
		bh.consume(starter.MatrixCalculationCudaKernel(n));
	}
	@Benchmark
    @Warmup(iterations = 1, time = 1)
    @Measurement(iterations = 1, time = 3)
    @BenchmarkMode(Mode.SingleShotTime)
	public void  MatrixCudaKernelShared (Blackhole bh){
		bh.consume(starter.MatrixCalculationCudaKernelShared(n));
	}
    public static void main(String[] args) throws Exception {
		Options options = new OptionsBuilder()
				.include(SetOfBenchmarks.class.getSimpleName())
				.warmupIterations(1)
				.measurementIterations(1)
				.forks(1)
				.build();
		new Runner(options).run();
	}
}
