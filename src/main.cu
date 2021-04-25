#include <iostream>
#include "tensor.hpp"

#define N (10)

__global__ void cuda_mul(float* a, float* b, float* out, int dim) {
    for(int i=0; i<dim; i++) {
        out[i] = a[i] * b[i];
    }
}

int main() {
    // tensor 1
    int shape_1[2] = {10, 20};
    int* matrix_1 = new int[200];
    for(int i=0; i<200; i++) {
        matrix_1[i] = 2;
    }
    Tensor& t1 = *(new Tensor(2, shape_1, matrix_1));

    // tensor 2
    int shape_2[2] = {20, 5};
    int* matrix_2 = new int[100];
    for(int i=0; i<100; i++) {
        matrix_2[i] = 3;
    }
    Tensor& t2 = *(new Tensor(2, shape_2, matrix_2));

    Tensor& t3 = t1 * t2;

    delete[] matrix_1;
    delete &t1;
    delete[] matrix_2;
    delete &t2;
    delete &t3;

    /********Check out the code below********/

    // host memory location
    float* a = (float*)malloc(sizeof(float)*N);
    float* b = (float*)malloc(sizeof(float)*N);
    float* out = (float*)malloc(sizeof(float)*N);

    // device memory location
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

    cuda_mul<<<1, 1>>>(d_a, d_b, d_out, N);

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

    return 0;
}