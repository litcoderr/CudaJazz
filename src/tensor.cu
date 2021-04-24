#include <iostream>
#include <stdexcept>
#include "tensor.hpp"

Tensor::Tensor(int dim, int* shape) {
    this->dim = dim;
    this->shape = new int[this->dim];
    for(int i=0; i<this->dim; i++) {
        // validate dimension size
        if(shape[i]<=0) {
            throw std::invalid_argument("Tensor dimension is less than or equal to 0");
        }
        this->shape[i] = shape[i];
    }
}

Tensor::~Tensor() {
    delete[] this->shape;
}

void Tensor::print_shape() {
    std::cout << "(";
    for(int i=0; i<this->dim; i++) {
        std::cout << this->shape[i];
        if(i!=this->dim-1) std::cout << ", ";
    }
    std::cout << ")\n";
}