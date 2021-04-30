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
    double matrix_1[200];
    for(int i=0; i<200; i++) {
        matrix_1[i] = 2;
    }
    Tensor& t1 = *(new Tensor(2, shape_1, matrix_1));
    
    // tensor 2
    int shape_2[2] = {20, 5};
    double matrix_2[100];
    for(int i=0; i<100; i++) {
        matrix_2[i] = 3;
    }
    Tensor& t2 = *(new Tensor(2, shape_2, matrix_2));

    // matrix multiplication
    Tensor& t3 = t1 * t2;

    t1.print();
    std::cout << std::endl;

    t2.print();
    std::cout << std::endl;

    t3.print();
    std::cout << std::endl;

    delete &t1;
    delete &t2;
    delete &t3;

    return 0;
}