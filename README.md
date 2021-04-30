# CudaJazz
> Learn cuda and implement linear algebra computation and deep learning framework

## Usage

### Tensor
Tensor is a multi-dimensional matrix that can be added, subtracted and multiplied. <br>

```cpp
#include "tensor.hpp"

// 1. define a shape
// in this case, it is a 3 dimensional shape
int shape[3] = {2, 10, 30};

// 2. define an array
double matrix_1[600];

// in this case, matrix is initialized as matrix of ones
for(int i=0; i<600; i++) {
    matrix_1[i] = 1;
}

// 3. allocate a tensor
// arguments are as follows. (int dimension, int* shape, double* matrix)
Tensor& t1 = *(new Tensor(3, shape_1, matrix_1));
```

#### Addition

```cpp
// t1 and t2 's type is Tensor&
// t1 and t2 's shape must be the same
Tensor& t3 = t1 + t2;
```

#### Subtraction

```cpp
// t1 and t2 's type is Tensor&
// t1 and t2 's shape must be the same
Tensor& t3 = t1 - t2;
```

#### Multiplication

```cpp
// t1 and t2 's type is Tensor&
// t1's last dimension and t2's last dimension must be the same
Tensor& t3 = t1 * t2;
```