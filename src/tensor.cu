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

    // Initialize matrix
    int size = this->get_size();
    this->matrix = new int[size];
    for(int i=0; i<size; i++) {
        this->matrix[i] = 0;
    }
}

Tensor::Tensor(int dim, int* shape, int* matrix): Tensor::Tensor(dim, shape) {
    // Update matrix to argument value
    int size = this->get_size();
    for(int i=0; i<size; i++) {
        this->matrix[i] = matrix[i];
    }
}

Tensor::~Tensor() {
    delete[] this->shape;
    delete[] this->matrix;
}

int Tensor::get_size() {
    int size = 0;
    for(int i=0; i<this->dim; i++) {
        size += this->shape[i];
    }
    return size;
}

void Tensor::print_shape() {
    std::cout << "(";
    for(int i=0; i<this->dim; i++) {
        std::cout << this->shape[i];
        if(i!=this->dim-1) std::cout << ", ";
    }
    std::cout << ")\n";
}