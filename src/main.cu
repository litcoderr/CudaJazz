#include <iostream>
#include "tensor.hpp"

int main() {
    // tensor 1
    int shape_1[2] = {20, 20};
    double matrix_1[400];
    for(int i=0; i<400; i++) {
        matrix_1[i] = 1;
    }
    Tensor& t1 = *(new Tensor(2, shape_1, matrix_1));
    
    // tensor 2
    int shape_2[2] = {20, 20};
    double matrix_2[400];
    for(int i=0; i<400; i++) {
        matrix_2[i] = 3;
    }
    Tensor& t2 = *(new Tensor(2, shape_2, matrix_2));

    // matrix multiplication
    Tensor& t3 = t1 * t2;

    // matrix addition
    Tensor& t4 = t1 + t2;

    t1.print_shape();
    t1.print();
    std::cout << std::endl;

    t2.print_shape();
    t2.print();
    std::cout << std::endl;

    t3.print_shape();
    t3.print();
    std::cout << std::endl;

    t4.print_shape();
    t4.print();
    std::cout << std::endl;

    delete &t1;
    delete &t2;
    delete &t3;
    delete &t4;

    return 0;
}