import numpy as np
from mytorch.tensor import Tensor

def test_relu_forward():
    x = Tensor(np.array([-2., -1., 0., 1., 2.]))
    y = x.relu()

    expected = np.array([0., 0., 0., 1., 2.])

    assert np.allclose(y.data, expected)
    
def test_relu_backward():
    x = Tensor(np.array([-2., -1., 0., 1., 2.]), requires_grad=True)

    y = x.relu()
    y.backward(np.ones_like(y.data))

    expected = np.array([0., 0., 0., 1., 1.])

    assert np.allclose(x.grad, expected)
    
def test_sigmoid_forward():
    x = Tensor(np.array([-1., 0., 1.]))

    y = x.sigmoid()

    expected = 1 / (1 + np.exp(-x.data))

    assert np.allclose(y.data, expected)
    
def test_sigmoid_backward():
    x = Tensor(np.array([-1., 0., 1.]), requires_grad=True)

    y = x.sigmoid()
    y.backward(np.ones_like(y.data))

    s = 1 / (1 + np.exp(-x.data))
    expected = s * (1 - s)

    assert np.allclose(x.grad, expected)
    
def test_softmax_forward():
    x = Tensor(np.array([1., 2., 3.]))

    y = x.softmax()

    e = np.exp(x.data - np.max(x.data))
    expected = e / e.sum()

    assert np.allclose(y.data, expected)
    
def test_softmax_sum_to_one():
    x = Tensor(np.random.randn(10))

    y = x.softmax()

    assert np.isclose(y.data.sum(), 1.0)
    
def test_softmax_batch():
    x = Tensor(np.random.randn(4, 5))

    y = x.softmax()

    assert np.allclose(y.data.sum(axis=-1), np.ones(4))
    
def test_log_softmax_forward():
    x = Tensor(np.array([1., 2., 3.]))

    y = x.log_softmax()

    shifted = x.data - np.max(x.data)
    logsumexp = np.log(np.exp(shifted).sum())

    expected = shifted - logsumexp

    assert np.allclose(y.data, expected)
    
def test_log_softmax_matches_softmax():
    x = Tensor(np.random.randn(8))

    y = x.log_softmax()

    assert np.allclose(np.exp(y.data), x.softmax().data)
    
def test_softmax_large_values():
    x = Tensor(np.array([1000., 1001., 1002.]))

    y = x.softmax()

    assert np.all(np.isfinite(y.data))
    assert np.isclose(y.data.sum(), 1.0)
    
def test_log_softmax_large_values():
    x = Tensor(np.array([1000., 1001., 1002.]))

    y = x.log_softmax()

    assert np.all(np.isfinite(y.data))
    
# backward

EPS = 1e-6
ATOL = 1e-5
RTOL = 1e-5

def numerical_gradient(f, x):
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index

        original = x[idx]

        x[idx] = original + EPS
        y1 = f(x)

        x[idx] = original - EPS
        y2 = f(x)

        x[idx] = original

        grad[idx] = (y1 - y2) / (2 * EPS)

        it.iternext()

    return grad

def test_relu_backward():
    x = np.random.randn(5, 4)

    def f(x_np):
        x = Tensor(x_np.copy(), requires_grad=True)
        return x.relu().sum().data

    numerical = numerical_gradient(f, x)

    x = Tensor(x.copy(), requires_grad=True)
    y = x.relu().sum()
    y.backward()

    assert np.allclose(x.grad, numerical, atol=ATOL, rtol=RTOL)
    
def test_sigmoid_backward():
    x = np.random.randn(5, 4)

    def f(x_np):
        x = Tensor(x_np.copy(), requires_grad=True)
        return x.sigmoid().sum().data

    numerical = numerical_gradient(f, x)

    x = Tensor(x.copy(), requires_grad=True)
    y = x.sigmoid().sum()
    y.backward()

    assert np.allclose(x.grad, numerical, atol=ATOL, rtol=RTOL)
    
def test_softmax_backward():
    x = np.random.randn(3, 5)
    w = np.random.randn(3, 5)

    def f(x_np):
        x = Tensor(x_np.copy(), requires_grad=True)
        return (x.softmax() * w).sum().data

    numerical = numerical_gradient(f, x)

    x = Tensor(x.copy(), requires_grad=True)
    y = (x.softmax() * w).sum()
    y.backward()

    assert np.allclose(x.grad, numerical, atol=ATOL, rtol=RTOL)
    
def test_log_softmax_backward():
    x = np.random.randn(3, 5)
    w = np.random.randn(3, 5)

    def f(x_np):
        x = Tensor(x_np.copy(), requires_grad=True)
        return (x.log_softmax() * w).sum().data

    numerical = numerical_gradient(f, x)

    x = Tensor(x.copy(), requires_grad=True)
    y = (x.log_softmax() * w).sum()
    y.backward()

    assert np.allclose(x.grad, numerical, atol=ATOL, rtol=RTOL)
    
def gradcheck(op, shape=(3, 4)):
    x_np = np.random.randn(*shape)
    w = np.random.randn(*shape)

    def f(x):
        t = Tensor(x.copy(), requires_grad=True)
        return (op(t) * w).sum().data

    numerical = numerical_gradient(f, x_np)

    t = Tensor(x_np.copy(), requires_grad=True)
    y = (op(t) * w).sum()
    y.backward()

    assert np.allclose(
        t.grad,
        numerical,
        atol=1e-5,
        rtol=1e-5,
    )
    
def test_relu_backward():
    gradcheck(lambda x: x.relu())

def test_sigmoid_backward():
    gradcheck(lambda x: x.sigmoid())

def test_softmax_backward():
    gradcheck(lambda x: x.softmax())

def test_log_softmax_backward():
    gradcheck(lambda x: x.log_softmax())
