#ifndef TENSOR_HPP
#define TENSOR_HPP

/** Parent Tensor class
*/
class Tensor {
public:
    /// dimension. ex) 10
    int dim;

    /// 1 dimensional shape
    int* shape;

    /** Constructor*/
    Tensor(int dim, int* shape);

    /** Deconstructor*/
    ~Tensor();


    /** print shape */
    void print_shape();
};

#endif