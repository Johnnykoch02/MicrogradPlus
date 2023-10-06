import torch # Using Torch to test differentiation
import numpy as np
from MicrogradPlus.microgradplus import NP_Value

def test_matmul():
    v1 = NP_Value.rnorm((5, 10))
    v2 =  NP_Value.rnorm((10, 2))
    x1 = (v1 @ v2).sum()
    x1.backward()
    
    v1_torch = torch.as_tensor(v1.data,)
    v2_torch = torch.as_tensor(v2.data,)
    v1_torch.requires_grad_()
    v2_torch.requires_grad_()
    x1_torch = (v1_torch @ v2_torch).sum()
    x1_torch.backward()
    
    assert np.isclose(v1.grad, v1_torch.grad.numpy()).all(), f'Grads do not match: {v1.grad} vs {v1_torch.grad.numpy()}'

def test_add():
    a = NP_Value.rnorm((5, 5))
    b = NP_Value.rnorm((5, 5))
    
    c = a + b
    c.backward()
    
    a_torch = torch.as_tensor(a.data)
    b_torch = torch.as_tensor(b.data)
    a_torch.requires_grad_()
    b_torch.requires_grad_()
    
    c_torch = a_torch + b_torch
    c_torch.backward(torch.ones_like(c_torch))
    
    assert np.isclose(a.grad, a_torch.grad.numpy()).all()
    assert np.isclose(b.grad, b_torch.grad.numpy()).all()

def test_mul():
    np.random.seed(0)
    torch.manual_seed(0)
    
    a = NP_Value.rnorm((5, 5))
    b = NP_Value.rnorm((5, 5))
    
    c = a * b
    c.backward()
    
    a_torch = torch.as_tensor(a.data)
    b_torch = torch.as_tensor(b.data)
    a_torch.requires_grad_()
    b_torch.requires_grad_()
    
    c_torch = a_torch * b_torch
    c_torch.backward(torch.ones_like(c_torch))
    
    assert np.isclose(a.grad, a_torch.grad.numpy()).all()
    assert np.isclose(b.grad, b_torch.grad.numpy()).all()

def test_div():
    np.random.seed(0)
    torch.manual_seed(0)
    
    a = NP_Value.rnorm((5, 5))
    b = NP_Value.rnorm((5, 5))
    
    c = a / b
    c.backward()
    
    a_torch = torch.as_tensor(a.data)
    b_torch = torch.as_tensor(b.data)
    a_torch.requires_grad_()
    b_torch.requires_grad_()
    
    c_torch = a_torch / b_torch
    c_torch.backward(torch.ones_like(c_torch))
    
    assert np.isclose(a.grad, a_torch.grad.numpy()).all()
    assert np.isclose(b.grad, b_torch.grad.numpy()).all()

def test_pow():
    base = 2
    exponent = 3
    v = NP_Value(np.random.randn(5, 5))
    x = v ** exponent
    x.backward()

    v_torch = torch.as_tensor(v.data,)
    v_torch.requires_grad_()
    x_torch = v_torch ** exponent
    x_torch.sum().backward()

    assert np.isclose(v.grad, v_torch.grad.numpy()).all(), f'Grads do not match: {v.grad} vs {v_torch.grad.numpy()}'

def test_sigmoid():
    v = NP_Value(np.random.randn(5, 5))
    x = v.sigmoid()
    x.backward()

    v_torch = torch.as_tensor(v.data,)
    v_torch.requires_grad_()
    x_torch = torch.sigmoid(v_torch)
    x_torch.sum().backward()

    assert np.isclose(v.grad, v_torch.grad.numpy()).all(), f'Grads do not match: {v.grad} vs {v_torch.grad.numpy()}'
    
def test_squeeze():
    v = NP_Value(np.random.randn(1, 5, 5, 1))
    x = v.squeeze(0)
    x.backward()
    
    v_torch = torch.as_tensor(v.data,)
    v_torch.requires_grad_()
    x_torch = torch.squeeze(v_torch, 0)
    x_torch.sum().backward()

    assert np.isclose(v.grad, v_torch.grad.numpy()).all(), f'Grads do not match: {v.grad} vs {v_torch.grad.numpy()}'

def test_sum():
    a = NP_Value.rnorm((5, 5))
    b = NP_Value.rnorm((5, 5))
    
    c = (a * b).sum() * 2
    
    c.backward()
    
    a_torch = torch.as_tensor(a.data)
    b_torch = torch.as_tensor(b.data)
    a_torch.requires_grad_()
    b_torch.requires_grad_()
    c_torch = (a_torch * b_torch).sum(dim=1) * 2
    
    c_torch.backward(torch.ones_like(c_torch))

    assert np.isclose(a.grad, a_torch.grad.numpy()).all()
    assert np.isclose(b.grad, b_torch.grad.numpy()).all()

def test_mean():
    a = NP_Value.rnorm((5, 5))
    mean_a = a.mean()
    mean_a.backward()
    
    a_torch = torch.as_tensor(a.data)
    a_torch.requires_grad_()
    mean_a_torch = torch.mean(a_torch)
    mean_a_torch.backward()
    assert np.isclose(a.grad, a_torch.grad.numpy()).all()

def test_log():
    a = NP_Value(np.random.rand(5, 5) + 1.0)
    log_a = a.log()
    b = log_a.mean()
    b.backward()
    
    a_torch = torch.as_tensor(a.data)
    a_torch.requires_grad_()
    log_a_torch = torch.log2(a_torch)
    b_torch = torch.mean(log_a_torch)
    b_torch.backward()
    
    assert np.isclose(a.grad, a_torch.grad.numpy()).all()

def run_tests():
    test_add()
    test_mul()
    test_div()
    test_matmul()
    test_pow()
    test_sigmoid()
    test_squeeze()
    test_sum()
    test_mean()
    test_log()
    print("All tests passed!")

if __name__ == "__main__":
    run_tests()
