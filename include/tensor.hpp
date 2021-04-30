#ifndef TENSOR_HPP
#define TENSOR_HPP

#define BLOCK_SIZE (32)

__global__ void cuda_mat_mul(double* m1, double* m2, double* m3, int N);

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
    int get_size() const;
    int get_lefthand_size() const;
    int get_righthand_size() const;

    /** print shape */
    void print_shape() const;

    /** print tensor */
    void print() const;

    friend Tensor& operator*(const Tensor& t1, const Tensor& t2);
};


#endif