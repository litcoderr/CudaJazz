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
    int shape_1[2] = {20, 30};
    double matrix_1[600];
    for(int i=0; i<600; i++) {
        matrix_1[i] = 1;
    }
    Tensor& t1 = *(new Tensor(2, shape_1, matrix_1));
    
    // tensor 2
    int shape_2[2] = {30, 5};
    double matrix_2[150];
    for(int i=0; i<150; i++) {
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