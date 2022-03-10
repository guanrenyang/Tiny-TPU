#include <torch/extension.h>

#include <iostream>
#include <vector>
#include <cstdlib>
#define N 100

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)


void MatMul_CUDA(const float *A_ptr, const float *B_ptr, float *C_ptr, const int b, const int m, const int k, const int n);


/*
const torch::Tensor &A: shape (b, m, k)
const torch::Tensor &B: shape (b, k, n)

MatrixMultiplication needs to calculate batch matrix multiplication
-------
return: torch::Tensor, shape (b, m, n)
*/
torch::Tensor MatrixMultiplication(const torch::Tensor &A, const torch::Tensor &B) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  AT_ASSERTM(A.size(0)==B.size(0), "Batch size for matrix A and B should be the same!");
  AT_ASSERTM(A.size(2)==B.size(1), "Matrix A and B should match for the k dimension!");

  const int b = A.size(0), m = A.size(1), k = A.size(2);
  const int n = B.size(2);
  auto C = torch::empty({b, m, n}, A.options());
  const float *A_ptr = (const float*)A.data_ptr();
  const float *B_ptr = (const float*)B.data_ptr();
  float *C_ptr = (float*)C.data_ptr();

  MatMul_CUDA(A_ptr, B_ptr, C_ptr, b, m, k, n);

  return C;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mymatmul", &MatrixMultiplication, "my Matrix Multiplication");
}
