#include <random>
#include <iostream>
#include <cuda.h>
#include <exception>

__global__ void add_matrix(float* A,float * B,float* C,int N){
    int index_x=blockIdx.x*blockDim.x+threadIdx.x;
    int index_y=blockIdx.y*blockDim.y+threadIdx.y;
    int index_array=index_y*N +index_x;
    C[index_array]=A[index_array]+B[index_array];
}

int main(){
    int N=16;
    int size=sizeof(float)*N*N;
    float h_A[N][N],h_B[N][N];
    float *d_A,*d_B,*d_C;
    for (int i=0;i<N;i++){
        for (int j=0;j<N;j++){
            float x=(float) rand()/RAND_MAX;
            h_A[i][j]=1-x;
            h_B[i][j]=x;
        };
    };
    cudaMalloc(&d_A,size);
    cudaMalloc(&d_B,size);
    cudaMalloc(&d_C,size);
    
    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    add_matrix<<<numBlocks,threadsPerBlock>>>(d_A,d_B,d_C,N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();
    
    float *h_result=(float*) malloc(size);
    cudaMemcpy(h_result,d_C,size,cudaMemcpyDeviceToHost);
    
    cudaFree(d_B);
    cudaFree(d_A);
    cudaFree(d_C);
    
    
    for (int i=0;i<N*N;i++){
        if (h_result[i]!=1){
            printf("%f ",h_result[i]);
        };
    }
    printf("\n");


}