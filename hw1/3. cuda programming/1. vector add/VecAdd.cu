#include <cuda.h>
#include <stdio.h>
#define BLOCKNUM 16

__global__
void vecAdd_kernel(float *A, float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n) C[i] = A[i] + B[i];
}

void vecAdd(float *A, float *B, float *C, int n) {
    float *dA, *dB, *dC;
    int size = n * sizeof(float);
    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);
    cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, size, cudaMemcpyHostToDevice);


    vecAdd_kernel<<<(n+BLOCKNUM-1)/BLOCKNUM, BLOCKNUM>>>(dA, dB, dC, n);
    cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);

    cudaFree(dA); cudaFree(dB); cudaFree(dC); 
}