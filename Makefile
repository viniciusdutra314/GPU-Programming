compile:
	g++ -x c++ image_inversion_cpu_and_gpu.cu -O3 -o cpu_inversion.elf
	/usr/local/cuda-12.4/bin/nvcc image_inversion_cpu_and_gpu.cu -O3 --extended-lambda -o gpu_inversion.elf
clean:
	rm cpu_inversion gpu_inversion