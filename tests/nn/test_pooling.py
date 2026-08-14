import numpy as np
import torch
import pytest

from mytorch.nn import (
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
)

from mytorch.tensor import Tensor


# ============================================================
# Helpers
# ============================================================

def assert_close(actual, expected, atol=1e-6, rtol=1e-5):
    np.testing.assert_allclose(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
    )


def mytorch_forward(pool, x):
    x_tensor = Tensor(x)
    y = pool(x_tensor)

    return y.data


def mytorch_backward(pool, x, grad_output):
    x_tensor = Tensor(x, requires_grad=True)

    y = pool(x_tensor)
    y.backward(grad_output)

    return x_tensor.grad


def torch_forward(pool, x):
    x_tensor = torch.tensor(
        x,
        dtype=torch.float64,
        requires_grad=True,
    )

    y = pool(x_tensor)

    return x_tensor, y.detach().numpy()


def torch_backward(pool, x, grad_output):
    x_tensor = torch.tensor(
        x,
        dtype=torch.float64,
        requires_grad=True,
    )

    y = pool(x_tensor)

    grad_output = torch.tensor(
        grad_output,
        dtype=torch.float64,
    )

    y.backward(grad_output)

    return x_tensor.grad.detach().numpy()


# ============================================================
# Configurations
# ============================================================

MAXPOOL_1D_CONFIGS = [
    # kernel, stride, padding, dilation
    (2, 2, 0, 1),
    (3, 1, 0, 1),
    (3, 2, 1, 1),
    (3, 1, 1, 2),
]

AVGPOOL_1D_CONFIGS = [
    # kernel, stride, padding
    (2, 2, 0),
    (3, 1, 0),
    (3, 2, 1),
]

MAXPOOL_2D_CONFIGS = [
    # kernel, stride, padding, dilation
    ((2, 2), (2, 2), (0, 0), (1, 1)),
    ((3, 3), (1, 1), (0, 0), (1, 1)),
    ((3, 3), (2, 2), (1, 1), (1, 1)),
    ((2, 3), (1, 2), (0, 1), (1, 1)),
    ((3, 2), (1, 1), (1, 0), (2, 1)),
]

AVGPOOL_2D_CONFIGS = [
    # kernel, stride, padding
    ((2, 2), (2, 2), (0, 0)),
    ((3, 3), (1, 1), (0, 0)),
    ((3, 3), (2, 2), (1, 1)),
    ((2, 3), (1, 2), (0, 1)),
]

MAXPOOL_3D_CONFIGS = [
    # kernel, stride, padding, dilation
    (
        (2, 2, 2),
        (2, 2, 2),
        (0, 0, 0),
        (1, 1, 1),
    ),
    (
        (3, 3, 3),
        (1, 1, 1),
        (0, 0, 0),
        (1, 1, 1),
    ),
    (
        (3, 3, 3),
        (2, 2, 2),
        (1, 1, 1),
        (1, 1, 1),
    ),
]

AVGPOOL_3D_CONFIGS = [
    # kernel, stride, padding
    (
        (2, 2, 2),
        (2, 2, 2),
        (0, 0, 0),
    ),
    (
        (3, 3, 3),
        (1, 1, 1),
        (0, 0, 0),
    ),
    (
        (3, 3, 3),
        (2, 2, 2),
        (1, 1, 1),
    ),
]


# ============================================================
# MaxPool1d
# ============================================================

@pytest.mark.parametrize(
    "kernel_size, stride, padding, dilation",
    MAXPOOL_1D_CONFIGS,
)
def test_maxpool1d_shape(
    kernel_size,
    stride,
    padding,
    dilation,
):
    x = np.random.randn(2, 3, 20)

    my_pool = MaxPool1d(
        kernel_size,
        stride,
        padding,
        dilation,
    )

    torch_pool = torch.nn.MaxPool1d(
        kernel_size,
        stride,
        padding,
        dilation,
    )

    _, expected = torch_forward(torch_pool, x)
    actual = mytorch_forward(my_pool, x)

    assert actual.shape == expected.shape


@pytest.mark.parametrize(
    "kernel_size, stride, padding, dilation",
    MAXPOOL_1D_CONFIGS,
)
def test_maxpool1d_forward(
    kernel_size,
    stride,
    padding,
    dilation,
):
    x = np.random.randn(2, 3, 20)

    my_pool = MaxPool1d(
        kernel_size,
        stride,
        padding,
        dilation,
    )

    torch_pool = torch.nn.MaxPool1d(
        kernel_size,
        stride,
        padding,
        dilation,
    )

    _, expected = torch_forward(torch_pool, x)
    actual = mytorch_forward(my_pool, x)

    assert_close(actual, expected)


def test_maxpool1d_backward():
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1

    x = np.random.randn(2, 3, 10)

    torch_pool = torch.nn.MaxPool1d(
        kernel_size,
        stride,
        padding,
        dilation,
    )

    _, y = torch_forward(torch_pool, x)

    grad_output = np.random.randn(*y.shape)

    expected = torch_backward(
        torch.nn.MaxPool1d(
            kernel_size,
            stride,
            padding,
            dilation,
        ),
        x,
        grad_output,
    )

    actual = mytorch_backward(
        MaxPool1d(
            kernel_size,
            stride,
            padding,
            dilation,
        ),
        x,
        grad_output,
    )

    assert_close(actual, expected)


# ============================================================
# AvgPool1d
# ============================================================

@pytest.mark.parametrize(
    "kernel_size, stride, padding",
    AVGPOOL_1D_CONFIGS,
)
def test_avgpool1d_shape(
    kernel_size,
    stride,
    padding,
):
    x = np.random.randn(2, 3, 20)

    my_pool = AvgPool1d(
        kernel_size,
        stride,
        padding,
    )

    torch_pool = torch.nn.AvgPool1d(
        kernel_size,
        stride,
        padding,
    )

    _, expected = torch_forward(torch_pool, x)
    actual = mytorch_forward(my_pool, x)

    assert actual.shape == expected.shape


@pytest.mark.parametrize(
    "kernel_size, stride, padding",
    AVGPOOL_1D_CONFIGS,
)
def test_avgpool1d_forward(
    kernel_size,
    stride,
    padding,
):
    x = np.random.randn(2, 3, 20)

    my_pool = AvgPool1d(
        kernel_size,
        stride,
        padding,
    )

    torch_pool = torch.nn.AvgPool1d(
        kernel_size,
        stride,
        padding,
    )

    _, expected = torch_forward(torch_pool, x)
    actual = mytorch_forward(my_pool, x)

    assert_close(actual, expected)


def test_avgpool1d_backward():
    kernel_size = 3
    stride = 1
    padding = 1

    x = np.random.randn(2, 3, 10)

    torch_pool = torch.nn.AvgPool1d(
        kernel_size,
        stride,
        padding,
    )

    _, y = torch_forward(torch_pool, x)

    grad_output = np.random.randn(*y.shape)

    expected = torch_backward(
        torch.nn.AvgPool1d(
            kernel_size,
            stride,
            padding,
        ),
        x,
        grad_output,
    )

    actual = mytorch_backward(
        AvgPool1d(
            kernel_size,
            stride,
            padding,
        ),
        x,
        grad_output,
    )

    assert_close(actual, expected)


# ============================================================
# MaxPool2d
# ============================================================

@pytest.mark.parametrize(
    "kernel_size, stride, padding, dilation",
    MAXPOOL_2D_CONFIGS,
)
def test_maxpool2d_shape(
    kernel_size,
    stride,
    padding,
    dilation,
):
    x = np.random.randn(2, 3, 16, 18)

    my_pool = MaxPool2d(
        kernel_size,
        stride,
        padding,
        dilation,
    )

    torch_pool = torch.nn.MaxPool2d(
        kernel_size,
        stride,
        padding,
        dilation,
    )

    _, expected = torch_forward(torch_pool, x)
    actual = mytorch_forward(my_pool, x)

    assert actual.shape == expected.shape


@pytest.mark.parametrize(
    "kernel_size, stride, padding, dilation",
    MAXPOOL_2D_CONFIGS,
)
def test_maxpool2d_forward(
    kernel_size,
    stride,
    padding,
    dilation,
):
    x = np.random.randn(2, 3, 16, 18)

    my_pool = MaxPool2d(
        kernel_size,
        stride,
        padding,
        dilation,
    )

    torch_pool = torch.nn.MaxPool2d(
        kernel_size,
        stride,
        padding,
        dilation,
    )

    _, expected = torch_forward(torch_pool, x)
    actual = mytorch_forward(my_pool, x)

    assert_close(actual, expected)


def test_maxpool2d_backward():
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)

    x = np.random.randn(2, 3, 10, 12)

    torch_pool = torch.nn.MaxPool2d(
        kernel_size,
        stride,
        padding,
        dilation,
    )

    _, y = torch_forward(torch_pool, x)

    grad_output = np.random.randn(*y.shape)

    expected = torch_backward(
        torch.nn.MaxPool2d(
            kernel_size,
            stride,
            padding,
            dilation,
        ),
        x,
        grad_output,
    )

    actual = mytorch_backward(
        MaxPool2d(
            kernel_size,
            stride,
            padding,
            dilation,
        ),
        x,
        grad_output,
    )

    assert_close(actual, expected)


# ============================================================
# AvgPool2d
# ============================================================

@pytest.mark.parametrize(
    "kernel_size, stride, padding",
    AVGPOOL_2D_CONFIGS,
)
def test_avgpool2d_shape(
    kernel_size,
    stride,
    padding,
):
    x = np.random.randn(2, 3, 16, 18)

    my_pool = AvgPool2d(
        kernel_size,
        stride,
        padding,
    )

    torch_pool = torch.nn.AvgPool2d(
        kernel_size,
        stride,
        padding,
    )

    _, expected = torch_forward(torch_pool, x)
    actual = mytorch_forward(my_pool, x)

    assert actual.shape == expected.shape


@pytest.mark.parametrize(
    "kernel_size, stride, padding",
    AVGPOOL_2D_CONFIGS,
)
def test_avgpool2d_forward(
    kernel_size,
    stride,
    padding,
):
    x = np.random.randn(2, 3, 16, 18)

    my_pool = AvgPool2d(
        kernel_size,
        stride,
        padding,
    )

    torch_pool = torch.nn.AvgPool2d(
        kernel_size,
        stride,
        padding,
    )

    _, expected = torch_forward(torch_pool, x)
    actual = mytorch_forward(my_pool, x)

    assert_close(actual, expected)


def test_avgpool2d_backward():
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)

    x = np.random.randn(2, 3, 10, 12)

    torch_pool = torch.nn.AvgPool2d(
        kernel_size,
        stride,
        padding,
    )

    _, y = torch_forward(torch_pool, x)

    grad_output = np.random.randn(*y.shape)

    expected = torch_backward(
        torch.nn.AvgPool2d(
            kernel_size,
            stride,
            padding,
        ),
        x,
        grad_output,
    )

    actual = mytorch_backward(
        AvgPool2d(
            kernel_size,
            stride,
            padding,
        ),
        x,
        grad_output,
    )

    assert_close(actual, expected)


# ============================================================
# MaxPool3d
# ============================================================

@pytest.mark.parametrize(
    "kernel_size, stride, padding, dilation",
    MAXPOOL_3D_CONFIGS,
)
def test_maxpool3d_shape(
    kernel_size,
    stride,
    padding,
    dilation,
):
    x = np.random.randn(2, 2, 10, 12, 14)

    my_pool = MaxPool3d(
        kernel_size,
        stride,
        padding,
        dilation,
    )

    torch_pool = torch.nn.MaxPool3d(
        kernel_size,
        stride,
        padding,
        dilation,
    )

    _, expected = torch_forward(torch_pool, x)
    actual = mytorch_forward(my_pool, x)

    assert actual.shape == expected.shape


@pytest.mark.parametrize(
    "kernel_size, stride, padding, dilation",
    MAXPOOL_3D_CONFIGS,
)
def test_maxpool3d_forward(
    kernel_size,
    stride,
    padding,
    dilation,
):
    x = np.random.randn(2, 2, 10, 12, 14)

    my_pool = MaxPool3d(
        kernel_size,
        stride,
        padding,
        dilation,
    )

    torch_pool = torch.nn.MaxPool3d(
        kernel_size,
        stride,
        padding,
        dilation,
    )

    _, expected = torch_forward(torch_pool, x)
    actual = mytorch_forward(my_pool, x)

    assert_close(actual, expected)


def test_maxpool3d_backward():
    kernel_size = (3, 3, 3)
    stride = (1, 1, 1)
    padding = (1, 1, 1)
    dilation = (1, 1, 1)

    x = np.random.randn(2, 2, 8, 9, 10)

    torch_pool = torch.nn.MaxPool3d(
        kernel_size,
        stride,
        padding,
        dilation,
    )

    _, y = torch_forward(torch_pool, x)

    grad_output = np.random.randn(*y.shape)

    expected = torch_backward(
        torch.nn.MaxPool3d(
            kernel_size,
            stride,
            padding,
            dilation,
        ),
        x,
        grad_output,
    )

    actual = mytorch_backward(
        MaxPool3d(
            kernel_size,
            stride,
            padding,
            dilation,
        ),
        x,
        grad_output,
    )

    assert_close(actual, expected)


# ============================================================
# AvgPool3d
# ============================================================

@pytest.mark.parametrize(
    "kernel_size, stride, padding",
    AVGPOOL_3D_CONFIGS,
)
def test_avgpool3d_shape(
    kernel_size,
    stride,
    padding,
):
    x = np.random.randn(2, 2, 10, 12, 14)

    my_pool = AvgPool3d(
        kernel_size,
        stride,
        padding,
    )

    torch_pool = torch.nn.AvgPool3d(
        kernel_size,
        stride,
        padding,
    )

    _, expected = torch_forward(torch_pool, x)
    actual = mytorch_forward(my_pool, x)

    assert actual.shape == expected.shape


@pytest.mark.parametrize(
    "kernel_size, stride, padding",
    AVGPOOL_3D_CONFIGS,
)
def test_avgpool3d_forward(
    kernel_size,
    stride,
    padding,
):
    x = np.random.randn(2, 2, 10, 12, 14)

    my_pool = AvgPool3d(
        kernel_size,
        stride,
        padding,
    )

    torch_pool = torch.nn.AvgPool3d(
        kernel_size,
        stride,
        padding,
    )

    _, expected = torch_forward(torch_pool, x)
    actual = mytorch_forward(my_pool, x)

    assert_close(actual, expected)


def test_avgpool3d_backward():
    kernel_size = (3, 3, 3)
    stride = (1, 1, 1)
    padding = (1, 1, 1)

    x = np.random.randn(2, 2, 8, 9, 10)

    torch_pool = torch.nn.AvgPool3d(
        kernel_size,
        stride,
        padding,
    )

    _, y = torch_forward(torch_pool, x)

    grad_output = np.random.randn(*y.shape)

    expected = torch_backward(
        torch.nn.AvgPool3d(
            kernel_size,
            stride,
            padding,
        ),
        x,
        grad_output,
    )

    actual = mytorch_backward(
        AvgPool3d(
            kernel_size,
            stride,
            padding,
        ),
        x,
        grad_output,
    )

    assert_close(actual, expected)


# ============================================================
# Overlapping windows
# ============================================================

def test_maxpool2d_overlapping_windows():
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1

    x = np.random.randn(1, 2, 7, 7)

    torch_pool = torch.nn.MaxPool2d(
        kernel_size,
        stride,
        padding,
        dilation,
    )

    _, expected = torch_forward(torch_pool, x)
    actual = mytorch_forward(
        MaxPool2d(
            kernel_size,
            stride,
            padding,
            dilation,
        ),
        x,
    )

    assert_close(actual, expected)

    grad_output = np.random.randn(*expected.shape)

    expected_dx = torch_backward(
        torch.nn.MaxPool2d(
            kernel_size,
            stride,
            padding,
            dilation,
        ),
        x,
        grad_output,
    )

    actual_dx = mytorch_backward(
        MaxPool2d(
            kernel_size,
            stride,
            padding,
            dilation,
        ),
        x,
        grad_output,
    )

    assert_close(actual_dx, expected_dx)


def test_avgpool2d_overlapping_windows():
    kernel_size = 3
    stride = 1
    padding = 0

    x = np.random.randn(1, 2, 7, 7)

    torch_pool = torch.nn.AvgPool2d(
        kernel_size,
        stride,
        padding,
    )

    _, expected = torch_forward(torch_pool, x)
    actual = mytorch_forward(
        AvgPool2d(
            kernel_size,
            stride,
            padding,
        ),
        x,
    )

    assert_close(actual, expected)

    grad_output = np.random.randn(*expected.shape)

    expected_dx = torch_backward(
        torch.nn.AvgPool2d(
            kernel_size,
            stride,
            padding,
        ),
        x,
        grad_output,
    )

    actual_dx = mytorch_backward(
        AvgPool2d(
            kernel_size,
            stride,
            padding,
        ),
        x,
        grad_output,
    )

    assert_close(actual_dx, expected_dx)


# ============================================================
# Known forward values
# ============================================================

def test_maxpool2d_known_output():
    x = np.array(
        [
            [
                [
                    [1, 3, 2, 4],
                    [5, 6, 1, 2],
                    [7, 2, 9, 1],
                    [3, 8, 4, 5],
                ]
            ]
        ],
        dtype=np.float64,
    )

    pool = MaxPool2d(
        kernel_size=2,
        stride=2,
    )

    actual = mytorch_forward(pool, x)

    expected = np.array(
        [
            [
                [
                    [6, 4],
                    [8, 9],
                ]
            ]
        ],
        dtype=np.float64,
    )

    assert_close(actual, expected)


def test_avgpool2d_known_output():
    x = np.array(
        [
            [
                [
                    [1, 3, 2, 4],
                    [5, 6, 1, 2],
                    [7, 2, 9, 1],
                    [3, 8, 4, 5],
                ]
            ]
        ],
        dtype=np.float64,
    )

    pool = AvgPool2d(
        kernel_size=2,
        stride=2,
    )

    actual = mytorch_forward(pool, x)

    expected = np.array(
        [
            [
                [
                    [3.75, 2.25],
                    [5.0, 4.75],
                ]
            ]
        ],
        dtype=np.float64,
    )

    assert_close(actual, expected)