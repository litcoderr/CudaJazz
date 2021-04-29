#ifndef TENSOR_HPP
#define TENSOR_HPP

/** Parent Tensor class
*/
class Tensor;
class Tensor {
public:
    /// dimension. ex) 10
    int dim;
    /// 1 dimensional shape
    int* shape;
    /// matrix
    double* matrix;

    /** Constructor
     * Default 0 Matrix
    */
    Tensor(int dim, int* shape);

    /** Constructor
     * Initialize matrix to argument value
    */
    Tensor(int dim, int* shape, double* matrix);

    /** Deconstructor*/
    ~Tensor();

    /** get matrix size
     * Use only if shape is valid 
    */
    int get_size();

    /** print shape */
    void print_shape();

    friend Tensor& operator*(const Tensor& t1, const Tensor& t2);
};


#endif