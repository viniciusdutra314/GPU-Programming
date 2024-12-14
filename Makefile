compile:
	g++ -O3 image_inversion_cpu.cpp -o cpu_inversion 
	/usr/local/cuda-12.4/bin/nvcc image_inversion_gpu.cu -O3 -arch=sm_61 -gencode=arch=compute_61,code=sm_61 -o gpu_inversion
clean:
	rm cpu_inversion gpu_inversion