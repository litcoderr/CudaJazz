#include <iostream>
#include "tensor.hpp"

#define N (10)

__global__ void cuda_hello(float* a, float* b, float* out, int dim) {
    for(int i=0; i<dim; i++) {
        out[i] = a[i] + b[i];
    }
}

int main() {
    Tensor1D* t1 = new Tensor1D(10);
    std::cout << "shape: " << t1->shape[0] << std::endl;

    /*
    // host memory
    float* a = (float*)malloc(sizeof(float)*N);
    float* b = (float*)malloc(sizeof(float)*N);
    float* out = (float*)malloc(sizeof(float)*N);

    float* d_a;
    float* d_b;
    float* d_out;

    // initializing array
    for(int i=0; i<N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
        out[i] = 0.0f;
    }

    // allocate device memory for a, b, out
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    // transfer to device memory
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    cuda_hello<<<1, 1>>>(d_a, d_b, d_out, N);

    // transfer to host memeory
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    for(int i=0; i<N; i++) {
        printf("%f ", out[i]);
    }

    // free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // free host memory
    free(a);
    free(b);
    free(out);

    */
    return 0;
}