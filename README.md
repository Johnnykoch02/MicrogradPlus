# MicrogradPlus 


## Introduction

Micrograd_Plus is an educational project aiming to provide a simple, yet extensible, NumPy-based automatic differentiation library. It is a modified version of the original [Micrograd](https://github.com/karpathy/micrograd) engine, with added functionalities for exploring the basics of deep learning frameworks. The goal of Micrograd_Plus is to extend Andrej Karpathy's micrograd implementation to provide a basis for vectorized auto gradients. The original Micrograd was built as a scalar-value gradient tool-kit and educational framework. This project aims to extend this functionality to Matricies and Vectors with the intention of introducing the Operations required to do so for calculating gradients in an auto-diff library.

This project was inspired by [Micrograd](https://github.com/karpathy/micrograd) and [Tinygrad](https://github.com/tinygrad/tinygrad) in hopes of expanding the gradient toolkit in an intermediate fashion. Tinygrad was inspired as a succesor to Micrograd, however does not provide a simplistic oversight in terms of demonstrating how the transition between scalar and vectorized auto gradients should be handled. This project aims to fill in the educational gaps between the two implementations. The idea of Tinygrad is to use ShapeTracker, where the shapes determine the gradients of the operands. In this implementation, it is clear where the shape tracker would be useful and how this can play into providing an optimized framework. 

If you liked this project, feel free to message me about it on [Twitter](https://twitter.com/jonathanzkoch) or check out my [website](https://jonathanzkoch.dev/home) for other avenues of connection.

Feel free to add any additional operations which I may have missed in this project, a pull request is all that is required!

## Features

- Scalar and matrix operations
- Backpropagation for automatic gradient calculation
- Functions for basic operations like addition, multiplication, division, and matrix multiplication
- Support for more advanced operations like sigmoid, logarithms, and power
- Element-wise operations and broadcasting

## Usage

Here is a simple example demonstrating the use of the library:

```python
from micrograd_plus import NP_Value

# Create two scalar values
a = NP_Value(2.0)
b = NP_Value(3.0)

# Perform operations
c = a + b
d = a * c

# Compute gradients
d.backward()

print(a.grad)  # Output should be the derivative of d with respect to a
```

## Core Components

### NP_Value

The `NP_Value` class is the heart of the library. Each `NP_Value` instance represents a differentiable scalar or tensor value. 

#### Operations Supported:

- `__add__`, `__sub__`, `__mul__`, `__div__`, `__matmul__`: Basic arithmetic operations
- `sigmoid`: Sigmoid activation function
- `log`: Logarithm
- `__pow__`: Power
- `sum`: Summation over a specific axis
- `unsqueeze`: Adds an extra dimension to the data
- `squeeze`: Removes dimensions of size 1 from the data

#### Properties:

- `data`: The actual data (NumPy array)
- `grad`: The gradient, initially set to zero

#### Methods:

- `backward()`: Computes the gradient with respect to all its ancestors in the computation graph

### Example:

```python
a = NP_Value([1.0, 2.0])
b = NP_Value([3.0, 4.0])
c = a + b

c.backward()
```

After the backward pass, gradients will be populated in `a.grad` and `b.grad`.

## Caveats and Limitations

- The current implementation does not handle all corner cases (e.g., higher-order gradients).
- Broadcasting is supported only in limited cases.

## Contribution

Feel free to contribute to the library by opening issues or pull requests.

## License

MIT License. See the LICENSE file for more details.
