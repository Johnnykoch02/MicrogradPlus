import numpy as np

class NP_Value:
        """
        Modified from: https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py   
        """
        def __init__(self, data, _children=(), ):
            if isinstance(data, (int, float)):
                data = [data]
            self.data = np.array(data, dtype=float)
            self.grad = np.zeros_like(self.data) # Initialize gradient to Zeros
            self._backward = lambda: None
            self._prev = set(_children)
        
        def sigmoid(self,):
            """
            Sigmoid(self) -> returns the sigmoid of the self.data
            """
            out = NP_Value(1 / (1 + np.exp(-self.data)), (self,))
            def _backward():
                self.grad += ( out.data * (1 - out.data)) * out.grad    
            out._backward = _backward
            return out
        
        def __add__(self, other):
            """
            Other: Union[NP_Value, Scalar/float]
            """
            other = other if isinstance(other, NP_Value) else NP_Value(other)
            out = NP_Value(self.data + other.data, (self, other), )
            def _backward():
                self.grad += out.grad
                other.grad += out.grad
            out._backward = _backward
            return out
        
        def __matmul__(self, other):
            """
            Self @ Other
            """
            other = other if isinstance(other, NP_Value) else NP_Value(other)
            out = NP_Value(self.data @ other.data, (self, other), )
            def _backward():
                self.grad += out.grad @ other.data.T
                other.grad += self.data.T @ out.grad
            out._backward = _backward
            return out
        
        def __mul__(self, other):
            """
            Self * Other 
            """
            other = other if isinstance(other, NP_Value) else NP_Value(other)
            out = NP_Value(self.data * other.data, (self, other),)
            
            def _backward():
                if np.isscalar(self.data) or self.data.shape == (1,):
                    self.grad += np.sum(other.data * out.grad)
                else:
                    self.grad += other.data * out.grad

                if np.isscalar(other.data) or other.data.shape == (1,):
                    other.grad += np.sum(self.data * out.grad)
                else:
                    other.grad += self.data * out.grad
            out._backward = _backward    
            return out
        
        def __div__(self, other):  # self / other
            """
            Self \div Other
            """
            other = other if isinstance(other, NP_Value) else NP_Value(other)
            out = NP_Value(self.data / other.data, (self, other),)
            
            def _backward():
                self.grad += (1 / other.data) * out.grad  # d(self/other)/dself = 1/b [Help from ChatGPT]
                other.grad += (-self.data / np.square(other.data)) * out.grad  # d(self/other)/dother = -a/b^2
            out._backward = _backward
            return out
        
        def __pow__(self, other):
            assert isinstance(other, (int, float)), "only supporting int/float powers for now"
            out = NP_Value(np.power(self.data, other), (self,), )
            def _backward():
                self.grad += (other * self.data**(other-1)) * out.grad
            out._backward = _backward
            return out
        
        def log(self, base=2):
            """
            log(Self)
            
            - base: base log performing calculation on
            
            """
            l_base = np.log(base)
            out = NP_Value((1/l_base) * np.log(self.data), (self,), )
            
            def _backward():
                self.grad += (1 / self.data * l_base) * out.grad
            out._backward = _backward
            return out
        
        def sum(self, dim=1):
            """
            Summation along dimension:

            - dim: dimension we will sum on
            
            """
            out = NP_Value(np.sum(self.data, axis=dim,), (self,))
            out._reduce_dim = dim
            def _backward():
                uncast = NP_Value(out.grad).unsqueeze(out._reduce_dim)
                self.grad += uncast.data * np.ones_like(self.data)
            out._backward = _backward
            return out
        
        def unsqueeze(self, dim):
            """
            Expand along a dimensioon
            """
            new_data = np.expand_dims(self.data, axis=dim)
            out = NP_Value(new_data, (self,))
            def _backward():
                self.grad += np.squeeze(out.grad, axis=dim)
            out._backward = _backward
            return out
        
        def squeeze(self, dim=None):
            """
            Reduce along a dimension
            """
            if dim is None:
                new_data = np.squeeze(self.data)
            else:
                new_data = np.squeeze(self.data, axis=dim)
            out = NP_Value(new_data, (self,))

            def _backward():
                if dim is None:
                    self.grad += np.reshape(out.grad, self.data.shape)
                else:
                    axis_shape = [1 if i == dim else j for i, j in enumerate(self.data.shape)]
                    self.grad += np.reshape(out.grad, axis_shape)
            out._backward = _backward
            return out
        
        @staticmethod
        def rnorm(size, mean=0, std=1.0):
            """
            Random Normal Distrobution with shape = Size:
            
            size: Shape of Tensor
            mean: mean of distrobution
            std: standard deviation of distrobution
            
            """
            out = NP_Value(data=np.random.normal(loc=mean, scale=std, size=size))
            return out
        
        def __neg__(self): # -self
            return self * -1
        def __radd__(self, other): # other + self
            return self + other
        def __sub__(self, other): # self - other
            return self + (-other)
        def __rsub__(self, other): # other - self
            return other + (-self)
        def __rmul__(self, other): # other * self
            return self * other
        def __truediv__(self, other): # self / other
            return self.__div__(other)
        def __rtruediv__(self, other): # other / self
            return other.__div__(self,)
        def __repr__(self):
            return f"NP_Value(data={self.data}, grad={self.grad})"
        
        def backward(self):
            """ Perform backpropagation to fill in Gradients """
            topo = []
            visited = set()
            def build_topo(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._prev:
                        build_topo(child)
                    topo.append(v)
            build_topo(self)
            self.grad = np.ones_like(self.data, dtype=float)
            [v._backward() for v in reversed(topo)]