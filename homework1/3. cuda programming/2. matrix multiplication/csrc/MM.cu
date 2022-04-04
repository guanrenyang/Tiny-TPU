#include <cuda.h>
#define BLOCKNUM 32

__global__
void MatMul_CUDA_Kernel(const float *A_ptr, const float *B_ptr, float *C_ptr, const int b, const int m, const int k, const int n) {
    // int x = threadIdx.x;
    // int y = threadIdx.y;
    // int z = threadIdx.z;
    int x = blockIdx.x;
    int y = blockIdx.y;
    int z = threadIdx.x;

    C_ptr[x*m*n+y*n+z] = 0;
    if(!(x<b&&y<m&&z<n))
    {
        return;
    }
    for(int i=0;i<k;++i){
        C_ptr[x*m*n+y*n+z] += A_ptr[x*m*k+y*k+i] * B_ptr[x*k*n+i*n+z];
    }    
}

void MatMul_CUDA(const float *A_ptr, const float *B_ptr, float *C_ptr, const int b, const int m, const int k, const int n) {
    
    float *dA, *dB, *dC;
    // compute memory size
    int size_A = b * m * k * sizeof(float);
    int size_B = b * k * n * sizeof(float);
    int size_C = b * m * n * sizeof(float);

    // malloc space on gpu
    cudaMalloc(&dA, size_A);
    cudaMalloc(&dB, size_B);
    cudaMalloc(&dC, size_C);

    // copy from host to device
    cudaMemcpy(dA, A_ptr, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B_ptr, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C_ptr, size_C, cudaMemcpyHostToDevice);

    
    // 一个block最多只能有1024个thread
    dim3 dimGrid(b, m, 1);
    int dimBlock = n;
    MatMul_CUDA_Kernel<<<dimGrid, dimBlock>>>(dA, dB, dC, b, m, k, n);

    cudaMemcpy(C_ptr, dC, size_C, cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}