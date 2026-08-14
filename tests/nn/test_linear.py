import numpy as np
from mytorch.tensor import Tensor
from mytorch.nn.modules.linear import Linear
from mytorch.nn.parameter import Parameter
import pytest

def test_weight_shape():
    layer = Linear(3, 5)

    assert layer.weight.shape == (5, 3)


def test_bias_shape():
    layer = Linear(3, 5)

    assert layer.bias.shape == (5,)


def test_no_bias():
    layer = Linear(3, 5, bias=False)

    assert layer.bias is None


def test_parameter_registration():
    layer = Linear(3, 5)

    params = list(layer.parameters())

    assert len(params) == 2
    assert any(p is layer.weight for p in params)
    assert any(p is layer.bias for p in params)
    

def test_parameter_registration_no_bias():
    layer = Linear(3, 5, bias=False)

    params = list(layer.parameters())

    assert len(params) == 1
    assert layer.weight in params


def test_named_parameters():
    layer = Linear(3, 5)

    names = dict(layer.named_parameters())

    assert set(names.keys()) == {"weight", "bias"}

    assert names["weight"] is layer.weight
    assert names["bias"] is layer.bias


def test_forward_shape_single():
    layer = Linear(3, 5)

    x = Tensor(np.random.randn(3))

    y = layer(x)

    assert y.shape == (5,)


def test_forward_shape_batch():
    layer = Linear(3, 5)

    x = Tensor(np.random.randn(8, 3))

    y = layer(x)

    assert y.shape == (8, 5)


@pytest.mark.parametrize("batch_size", [1, 2, 8, 32])
def test_forward_batch_sizes(batch_size):
    layer = Linear(4, 7)

    x = Tensor(np.random.randn(batch_size, 4))

    y = layer(x)

    assert y.shape == (batch_size, 7)


def test_forward_known_weights():
    layer = Linear(2, 2)

    layer.weight.data = np.array([
        [1., 2.],
        [3., 4.]
    ])

    layer.bias.data = np.array([
        5.,
        6.
    ])

    x = Tensor(np.array([
        [10., 20.]
    ]))

    y = layer(x)

    expected = np.array([
        [
            10 * 1 + 20 * 2 + 5,
            10 * 3 + 20 * 4 + 6
        ]
    ])

    np.testing.assert_allclose(y.data, expected)


def test_forward_no_bias():
    layer = Linear(2, 2, bias=False)

    layer.weight.data = np.array([
        [1., 2.],
        [3., 4.]
    ])

    x = Tensor(np.array([
        [10., 20.]
    ]))

    y = layer(x)

    expected = np.array([
        [
            10 * 1 + 20 * 2,
            10 * 3 + 20 * 4
        ]
    ])

    np.testing.assert_allclose(y.data, expected)


def test_weight_is_parameter():
    layer = Linear(3, 5)

    assert isinstance(layer.weight, Parameter)


def test_bias_is_parameter():
    layer = Linear(3, 5)

    assert isinstance(layer.bias, Parameter)


def test_train():
    layer = Linear(3, 5)

    layer.eval()
    assert not layer.training

    layer.train()
    assert layer.training


def test_eval():
    layer = Linear(3, 5)

    layer.eval()

    assert not layer.training


def test_zero_grad():
    layer = Linear(3, 5)

    layer.weight.grad = np.random.randn(*layer.weight.shape)
    layer.bias.grad = np.random.randn(*layer.bias.shape)

    layer.zero_grad()

    assert layer.weight.grad is None
    assert layer.bias.grad is None
    
import numpy as np

from mytorch import Tensor
from mytorch import nn


# ============================================================
# LazyLinear Tests
# ============================================================

def test_lazylinear_initialization():
    """LazyLinear should not know in_features before first forward."""

    layer = nn.LazyLinear(10)

    assert layer.in_features is None


def test_lazylinear_infers_in_features():
    """LazyLinear should infer in_features from the first input."""

    layer = nn.LazyLinear(10)

    x = Tensor(np.random.randn(4, 20))
    y = layer(x)

    assert layer.in_features == 20
    assert y.shape == (4, 10)


def test_lazylinear_parameter_shapes():
    """Weight and bias should have the correct shapes after initialization."""

    layer = nn.LazyLinear(10)

    x = Tensor(np.random.randn(4, 20))
    layer(x)

    assert layer.weight.shape == (10, 20)
    assert layer.bias.shape == (10,)


def test_lazylinear_reuses_inferred_shape():
    """The layer should keep the inferred input size on subsequent forwards."""

    layer = nn.LazyLinear(10)

    x1 = Tensor(np.random.randn(4, 20))
    x2 = Tensor(np.random.randn(8, 20))

    y1 = layer(x1)
    y2 = layer(x2)

    assert layer.in_features == 20

    assert y1.shape == (4, 10)
    assert y2.shape == (8, 10)


def test_lazylinear_rejects_different_in_features():
    """After initialization, a different feature size should raise an error."""

    layer = nn.LazyLinear(10)

    x1 = Tensor(np.random.randn(4, 20))
    layer(x1)

    x2 = Tensor(np.random.randn(4, 30))

    try:
        layer(x2)
        assert False, "LazyLinear should reject a different in_features"
    except (ValueError, RuntimeError):
        pass


def test_lazylinear_forward():
    """Check forward against the expected matrix multiplication."""

    layer = nn.LazyLinear(3)

    x = Tensor(
        np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ])
    )

    y = layer(x)

    expected = x.data @ layer.weight.data.T + layer.bias.data

    np.testing.assert_allclose(
        y.data,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_lazylinear_output_shape():
    """Check output shape for different batch sizes."""

    layer = nn.LazyLinear(7)

    for batch_size in [1, 2, 16, 32]:

        x = Tensor(np.random.randn(batch_size, 15))
        y = layer(x)

        assert y.shape == (batch_size, 7)


def test_lazylinear_zero_batch_dimension():
    """Check that batch dimension is handled correctly."""

    layer = nn.LazyLinear(5)

    x = Tensor(np.random.randn(0, 10))
    y = layer(x)

    assert y.shape == (0, 5)


def test_lazylinear_backward():
    """Check that gradients are created after backward."""

    layer = nn.LazyLinear(3)

    x = Tensor(np.random.randn(4, 5), requires_grad=True)

    y = layer(x)

    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None


def test_lazylinear_gradient_shapes():
    """Check gradient shapes."""

    layer = nn.LazyLinear(3)

    x = Tensor(np.random.randn(4, 5), requires_grad=True)

    y = layer(x)
    loss = y.sum()
    loss.backward()

    assert x.grad.shape == x.shape
    assert layer.weight.grad.shape == layer.weight.shape
    assert layer.bias.grad.shape == layer.bias.shape


def test_lazylinear_parameters_registered_after_forward():
    """LazyLinear parameters should become registered after initialization."""

    layer = nn.LazyLinear(10)

    # Before forward
    assert layer.in_features is None

    x = Tensor(np.random.randn(4, 20))
    layer(x)

    params = list(layer.parameters())

    assert len(params) == 2
    assert any(p is layer.weight for p in params)
    assert any(p is layer.bias for p in params)


def test_lazylinear_different_output_sizes():
    """Different out_features should produce different output dimensions."""

    for out_features in [1, 5, 10, 32, 128]:

        layer = nn.LazyLinear(out_features)

        x = Tensor(np.random.randn(4, 20))
        y = layer(x)

        assert y.shape == (4, out_features)


def test_lazylinear_with_flatten():
    """
    Test the actual use case:
        Conv -> Flatten -> LazyLinear
    """

    model = nn.Sequential(
        nn.Flatten(),
        nn.LazyLinear(32),
        nn.ReLU(),
        nn.Linear(32, 10),
    )

    x = Tensor(np.random.randn(8, 64, 24, 24))

    y = model(x)

    assert y.shape == (8, 10)

    # 64 * 24 * 24
    assert model[1].in_features == 64 * 24 * 24


def test_lazylinear_multiple_input_shapes():
    """
    The same LazyLinear should work with different batch sizes
    as long as the feature dimension stays the same.
    """

    layer = nn.LazyLinear(16)

    for batch_size in [1, 4, 8, 32]:

        x = Tensor(np.random.randn(batch_size, 50))
        y = layer(x)

        assert y.shape == (batch_size, 16)
        assert layer.in_features == 50
