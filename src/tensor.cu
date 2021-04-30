#include <iostream>
#include <cmath>
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
    this->matrix = new double[size];
    for(int i=0; i<size; i++) {
        this->matrix[i] = 0;
    }
}

Tensor::Tensor(int dim, int* shape, double* matrix): Tensor::Tensor(dim, shape) {
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

int Tensor::get_size() const {
    int size = 1;
    for(int i=0; i<this->dim; i++) {
        size *= this->shape[i];
    }
    return size;
}

int Tensor::get_lefthand_size() const {
    if(this->dim < 2) {
        throw std::length_error("Non-matrix tensor does not have lefthand size");
    }
    int size = 1;
    for(int i=0;i<dim-1;i++) {
        size *= this->shape[i];
    }
    return size;
}

int Tensor::get_righthand_size() const {
    if(this->dim < 2) {
        throw std::length_error("Non-matrix tensor does not have righthand size");
    }
    int size = 1;
    for(int i=1;i<dim;i++) {
        size *= this->shape[i];
    }
    return size;
}

void Tensor::print_shape() const {
    std::cout << "(";
    for(int i=0; i<this->dim; i++) {
        std::cout << this->shape[i];
        if(i!=this->dim-1) std::cout << ", ";
    }
    std::cout << ")\n";
}

void Tensor::print() const {

}

/**************** Operator Overloading ******************/
Tensor& operator*(const Tensor& t1, const Tensor& t2) {
    // throw invalid dimension for matrix multiplication
    if(t1.dim<2 || t2.dim<2) {
        throw std::length_error("Non-matrix tensors cannot be mutiplied");
    }
    if(t1.shape[t1.dim-1]!=t2.shape[0]) {
        throw std::length_error("Invalid dimension for matrix multiplication");
    }

    // init shape
    int dim_3 = t1.dim + t2.dim -2;
    int* shape = new int[dim_3];
    for(int i=0; i< t1.dim-1; i++) {
        shape[i] = t1.shape[i];
    }
    for(int i=1; i< t2.dim; i++) {
        shape[t1.dim+i-2] = t2.shape[i];
    }

    Tensor& t3 = *(new Tensor(dim_3, shape));

    /*--------------- 1. Compute matrix ---------------*/
    // host memory (t1.matrix, t2.matrix, t3.matrix)
    // allocate device memory
    double* m1;
    double* m2;
    double* m3;

    cudaMalloc((void**)&m1, sizeof(double) * t1.get_size());
    cudaMalloc((void**)&m2, sizeof(double) * t2.get_size());
    cudaMalloc((void**)&m3, sizeof(double) * t3.get_size());

    // copy t1, t2's matrix to device
    cudaMemcpy(m1, t1.matrix, sizeof(double) * t1.get_size(), cudaMemcpyHostToDevice);
    cudaMemcpy(m2, t2.matrix, sizeof(double) * t2.get_size(), cudaMemcpyHostToDevice);

    // compute
    dim3 blocksPerGrid((int)std::ceil((double)t1.get_lefthand_size()/BLOCK_SIZE),
                       (int)std::ceil((double)t2.get_righthand_size()/BLOCK_SIZE));
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);

    cuda_mat_mul<<<blocksPerGrid, threadsPerBlock>>>(m1, m2, m3, t2.shape[0]);

    /*--------------- 2. Update Matrix ---------------*/
    cudaMemcpy(t3.matrix, m3, sizeof(double) * t3.get_size(), cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(m1);
    cudaFree(m2);
    cudaFree(m3);
    
    delete shape;
    return t3;
}


/****************  Cuda Kernel Definition ******************/

__global__ void cuda_mat_mul(double* m1, double* m2, double* m3, int N) {
    // TODO define cuda kernel for matrix multiplication
}
