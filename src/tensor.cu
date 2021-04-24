#include <iostream>
#include <stdexcept>
#include "tensor.hpp"

Tensor1D::Tensor1D(int d1) {
    this->dim = 1;
    this->shape = new int[this->dim];
    this->shape[0] = d1;

    // validate dimension size
    // TODO validate from parent Tensor class constructor
    if(d1<=0) {
        throw std::invalid_argument("Tensor1D dimension is less than or equal to 0");
    }
}