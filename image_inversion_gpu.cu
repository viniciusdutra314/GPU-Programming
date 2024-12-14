#include <iostream>
#include <cmath>
#include <cuda.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void invert_pixel(unsigned char* pixel){
    auto index=blockDim.x*blockIdx.x+threadIdx.x;
    pixel[index]=255-pixel[index];
}

int main() {
    const char* filename = "input_example.jpg";
    int width, height, channels;
    unsigned char* h_image = stbi_load(filename, &width, &height, &channels, 0);

    if (!h_image) {
        std::cerr << "Failed to load image: " << stbi_failure_reason() << std::endl;
        return 1;
    }
    int memory_size=sizeof(char)*width*height*channels;
    unsigned char *d_image;

    cudaMalloc(&d_image,memory_size);
    cudaMemcpy(d_image,h_image,memory_size,cudaMemcpyHostToDevice);

    int threads_per_block=64;
    int num_of_blocks=ceil((float)memory_size/threads_per_block);

    invert_pixel<<<num_of_blocks,threads_per_block>>>(d_image);
    cudaDeviceSynchronize();
    auto err =cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Error: " << cudaGetErrorString(err) << std::endl;
    }
    
    auto h_result=(unsigned char*) malloc(memory_size);
    cudaMemcpy(h_result,d_image,memory_size,cudaMemcpyDeviceToHost);
    stbi_write_jpg("output_example.jpg",width, height, 3, h_result, 100);

    cudaFree(d_image);
    free(h_image);
    free(h_result);
}