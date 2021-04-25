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
    /// matrix
    int* matrix;

    /** Constructor
     * Default 0 Matrix
    */
    Tensor(int dim, int* shape);

    /** Constructor
     * Initialize matrix to argument value
    */
    Tensor(int dim, int* shape, int* matrix);

    /** Deconstructor*/
    ~Tensor();

    /** get matrix size
     * Use only if shape is valid 
    */
    int get_size();

    /** print shape */
    void print_shape();
};

#endif