#include <cuda.h>
#include <stdexcept>
#include <stdio.h>
#include <chrono>
#define TILE_WIDTH 32

//A x B = C
__global__ void matrix_multiply(float* A,float* B,float* C, int size){
    __shared__ float A_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_tile[TILE_WIDTH][TILE_WIDTH];

    //A[i][j] is on the i-row and j-column
    int i=blockIdx.y*TILE_WIDTH+threadIdx.y;
    int j=blockIdx.x*TILE_WIDTH+threadIdx.x;
    float c_ij=0;

    // 2 TILE_WIDTH x size (elements to cache)
    // 2 TILE_WIDTH^2 cached at a time
    // its needed size/tile_width phases
    for (int phase=0;phase<size/TILE_WIDTH;++phase){
        int offset=phase*TILE_WIDTH;
        //getting A[i][k] k varying 
        A_tile[threadIdx.y][threadIdx.x]=A[size*i +offset+threadIdx.x];
        //getting B[k][j] k varying like a sliding window
        B_tile[threadIdx.y][threadIdx.x]=B[size*(offset+threadIdx.y)+j];
        
        __syncthreads();
        //(A x B)ij=sum_k Aik x B_kj
        // size/TILE_WIDTH x TILE_WDITH goes through all the elements
        for (int k=0;k<TILE_WIDTH;++k){
            c_ij+=A_tile[threadIdx.y][k]*B_tile[k][threadIdx.x];
            
        }
        __syncthreads();
    }
    C[i*size+j]=c_ij;
}

int main(){
    int size=26*TILE_WIDTH;
    
    float h_A[size][size],h_B[size][size],h_C[size][size];
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            h_A[i][j]=(float) rand()/RAND_MAX;
            h_B[i][j]=(float) rand()/RAND_MAX;
        }
    }

    // CPU implementation with timing
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (size_t i=0;i<size;i++){
        for (size_t j=0;j<size;j++){
            h_C[i][j]=0;
            for (size_t k=0;k<size;k++){
                h_C[i][j]+=h_A[i][k]*h_B[k][j];
            }
        }
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
    printf("CPU Time: %f milliseconds\n", 1'000*cpu_duration.count());
    
    float * d_A,*d_B,*d_C;
    int memory_size=sizeof(float)*size*size;
    cudaMalloc(&d_A,memory_size);
    cudaMalloc(&d_B,memory_size);
    cudaMalloc(&d_C,memory_size);

    cudaMemcpy(d_A,h_A,memory_size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,memory_size,cudaMemcpyHostToDevice);
    dim3 numBlocks (size/TILE_WIDTH,size/TILE_WIDTH,1);
    dim3 threadsPerBlock (TILE_WIDTH,TILE_WIDTH,1);
    
    // GPU implementation with timing
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);
    
    matrix_multiply<<<numBlocks,threadsPerBlock>>>(d_A,d_B,d_C,size);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    
    float gpu_duration = 0;
    cudaEventElapsedTime(&gpu_duration, start_gpu, stop_gpu);
    printf("GPU Time: %f milliseconds\n", gpu_duration);
    
    cudaDeviceSynchronize();
    auto h_C_from_GPU=(float*) malloc(memory_size);
    cudaMemcpy(h_C_from_GPU,d_C,memory_size,cudaMemcpyDeviceToHost);
    
    //verifying results
    int counter=0;
    for (size_t i=0;i<size;i++){
        for (size_t j=0;j<size;j++){
            if (abs(h_C[i][j]-h_C_from_GPU[i*size+j])>1e-4){
                printf("%f %f \n",h_C[i][j],h_C_from_GPU[i*size+j]);
                counter++;
                //throw std::logic_error("Methods don't produce the same result");
            }
        }
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);
}