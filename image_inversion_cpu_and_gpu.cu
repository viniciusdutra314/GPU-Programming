#ifndef CUDA_MACROS
#define CUDA_MACROS
    #ifndef __CUDACC__
        #define HOST
        #define DEVICE
        #else
        #define HOST __host__
        #define DEVICE __device
    #endif
#endif

#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifdef __CUDACC__
    #include <thrust/device_vector.h>
    #include <thrust/host_vector.h>
    #include <thrust/for_each.h>
    #include <thrust/execution_policy.h>
#endif 


int main() {
    const char* filename = "input_example.jpg";
    int width, height, channels;
    unsigned char* h_img_ptr = stbi_load(filename, &width, &height, &channels, 0);
    
    if (!h_img_ptr) {
        std::cerr << "Failed to load image: " << stbi_failure_reason() << std::endl;
        return 1;
    }
    
    #ifdef __CUDACC__
        auto h_img_vec = thrust::host_vector<uint8_t>(h_img_ptr, h_img_ptr + width * height * channels);
        thrust::device_vector<uint8_t> d_img = h_img_vec;
        auto invert = [] __device__ (uint8_t& x) { x = 255 - x; };
        thrust::for_each(thrust::device, d_img.begin(), d_img.end(), invert);
        thrust::copy(d_img.begin(), d_img.end(), h_img_ptr);
        stbi_write_jpg("output_gpu_example.jpg", width, height, channels, h_img_ptr, 100);
    #else
        for (int i=0;i<width*height*channels;i++){
            *(h_img_ptr+i)=255-*(h_img_ptr+i);
        }
        stbi_write_jpg("output_cpu_example.jpg", width, height, channels, h_img_ptr, 100);
    #endif

    stbi_image_free(h_img_ptr);
    return 0;
}