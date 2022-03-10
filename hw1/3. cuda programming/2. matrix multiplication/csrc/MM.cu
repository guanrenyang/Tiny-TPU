#include <cuda.h>
#define BLOCKNUM 32

__global__
void MatMul_CUDA_Kernel(const float *A_ptr, const float *B_ptr, float *C_ptr, const int b, const int m, const int k, const int n) {

}

void MatMul_CUDA(const float *A_ptr, const float *B_ptr, float *C_ptr, const int b, const int m, const int k, const int n) {
    
}