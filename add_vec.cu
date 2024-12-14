#include <random>
#include <iostream>
#include <cuda.h>

__global__ void add_vec(float* v,float * u,float* result){
    auto index=threadIdx.x;
    result[index]=v[index]+u[index];
}

int main(){
    int N=100;
    int size=sizeof(float)*N;
    float h_v[N],h_u[N];
    float *d_v,*d_u,*d_result;
    float *h_result=(float*) malloc(size);
    for (int i=0;i<N;i++){
        h_v[i]=0.75*i;
        h_u[i]=0.25*i;
    }
    cudaMalloc(&d_v,size);
    cudaMalloc(&d_u,size);
    cudaMalloc(&d_result,size);
    
    cudaMemcpy(d_v,h_v,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_u,h_u,size,cudaMemcpyHostToDevice);
    add_vec<<<1,N>>>(d_v,d_u,d_result);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();
    

    cudaMemcpy(h_result,d_result,size,cudaMemcpyDeviceToHost);
    
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_result);
    
    
    for (int i=0;i<N;i++){
        printf("%f ",h_result[i]);
    }
    printf("\n");


}