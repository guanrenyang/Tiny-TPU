#include <iostream>
#include <vector>
#define N 100

void vecAdd(float *A, float *B, float *C, int n);

int main() {
  std::vector<float> A(N, 2), B(N, -56), C(N, 100);

  vecAdd(A.data(), B.data(), C.data(), N);

  for (int i=0; i<N; ++i)
    std::cout << C[i] << " ";
  std::cout << std::endl;
}