#ifndef TENSOR_HPP
#define TENSOR_HPP

/** Parent Tensor class
*/
class Tensor {
public:
    
};

class Tensor1D: public Tensor {
public:
    /// dimension. ex) 10
    int dim;

    /// 1 dimensional shape
    int* shape;

    /** Constructor*/
    Tensor1D(int dim);
};

#endif