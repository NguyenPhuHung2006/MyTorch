# tests/nn/test_conv.py

import numpy as np
import pytest
import torch

from mytorch import Tensor
from mytorch.nn.modules.conv import Conv1d, Conv2d, Conv3d


# ============================================================
# Configuration
# ============================================================

ATOL = 1e-4
RTOL = 1e-4


# ============================================================
# Helpers
# ============================================================

def assert_array_equal(
    actual,
    expected,
    *,
    name="array",
    atol=ATOL,
    rtol=RTOL,
):
    """
    Compare two arrays and provide useful information when
    the comparison fails.
    """

    actual = np.asarray(actual)
    expected = np.asarray(expected)

    assert actual.shape == expected.shape, (
        f"{name} shape mismatch:\n"
        f"  MyTorch : {actual.shape}\n"
        f"  PyTorch : {expected.shape}"
    )

    np.testing.assert_allclose(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
        err_msg=f"{name} values do not match",
    )


def copy_parameters(mytorch_conv, torch_conv):
    """
    Copy PyTorch parameters into the MyTorch convolution.

    This guarantees that both implementations receive
    exactly the same weights and bias.
    """

    mytorch_conv.weight.data[...] = (
        torch_conv.weight.detach().numpy()
    )

    if mytorch_conv.bias is not None:
        mytorch_conv.bias.data[...] = (
            torch_conv.bias.detach().numpy()
        )


def compare_forward(
    mytorch_conv,
    torch_conv,
    x,
):
    """
    Compare forward output from MyTorch and PyTorch.
    """

    torch_x = torch.tensor(
        x,
        dtype=torch.float32,
        requires_grad=True,
    )

    mytorch_x = Tensor(
        x.copy(),
        requires_grad=True,
    )

    # Forward
    torch_output = torch_conv(torch_x)
    mytorch_output = mytorch_conv(mytorch_x)

    # Check shape
    assert mytorch_output.shape == tuple(
        torch_output.shape
    ), (
        f"Output shape mismatch:\n"
        f"  MyTorch : {mytorch_output.shape}\n"
        f"  PyTorch : {tuple(torch_output.shape)}"
    )

    # Check values
    assert_array_equal(
        mytorch_output.data,
        torch_output.detach().numpy(),
        name="forward output",
    )

    return mytorch_x, torch_x, mytorch_output, torch_output


def compare_backward(
    mytorch_x,
    torch_x,
    mytorch_output,
    torch_output,
    mytorch_conv,
    torch_conv,
    *,
    grad_output=None,
):
    """
    Compare backward gradients:

        dL/dx
        dL/dweight
        dL/dbias

    between MyTorch and PyTorch.

    We use the SAME upstream gradient for both frameworks.
    """

    # --------------------------------------------------------
    # Create an upstream gradient
    #
    # Instead of using loss = output.sum(), using a random
    # grad_output tests the backward implementation more
    # thoroughly.
    # --------------------------------------------------------

    if grad_output is None:
        rng = np.random.default_rng(12345)

        grad_output = rng.normal(
            size=mytorch_output.shape
        ).astype(np.float32)

    # --------------------------------------------------------
    # PyTorch backward
    # --------------------------------------------------------

    torch_output.backward(
        torch.tensor(
            grad_output,
            dtype=torch.float32,
        )
    )

    # --------------------------------------------------------
    # MyTorch backward
    # --------------------------------------------------------

    mytorch_output.backward(
        grad_output
    )

    # --------------------------------------------------------
    # Check gradient with respect to input
    # --------------------------------------------------------

    assert mytorch_x.grad is not None, (
        "MyTorch input gradient is None."
    )

    assert torch_x.grad is not None, (
        "PyTorch input gradient is None."
    )

    assert_array_equal(
        mytorch_x.grad,
        torch_x.grad.detach().numpy(),
        name="input gradient (dL/dx)",
    )

    # --------------------------------------------------------
    # Check gradient with respect to weight
    # --------------------------------------------------------

    assert mytorch_conv.weight.grad is not None, (
        "MyTorch weight gradient is None."
    )

    assert torch_conv.weight.grad is not None, (
        "PyTorch weight gradient is None."
    )

    assert_array_equal(
        mytorch_conv.weight.grad,
        torch_conv.weight.grad.detach().numpy(),
        name="weight gradient (dL/dweight)",
    )

    # --------------------------------------------------------
    # Check gradient with respect to bias
    # --------------------------------------------------------

    if mytorch_conv.bias is not None:

        assert mytorch_conv.bias.grad is not None, (
            "MyTorch bias gradient is None."
        )

        assert torch_conv.bias.grad is not None, (
            "PyTorch bias gradient is None."
        )

        assert_array_equal(
            mytorch_conv.bias.grad,
            torch_conv.bias.grad.detach().numpy(),
            name="bias gradient (dL/dbias)",
        )


def run_conv_test(
    mytorch_conv,
    torch_conv,
    x,
):
    """
    Run the complete forward + backward comparison.
    """

    # Make parameters identical
    copy_parameters(
        mytorch_conv,
        torch_conv,
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    (
        mytorch_x,
        torch_x,
        mytorch_output,
        torch_output,
    ) = compare_forward(
        mytorch_conv,
        torch_conv,
        x,
    )

    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------

    compare_backward(
        mytorch_x,
        torch_x,
        mytorch_output,
        torch_output,
        mytorch_conv,
        torch_conv,
    )


# ============================================================
# Conv1D
# ============================================================

def test_conv1d_basic():
    """
    Basic Conv1D test.

    Tests:
        - output shape
        - output values
        - dL/dx
        - dL/dweight
        - dL/dbias
    """

    rng = np.random.default_rng(0)

    x = rng.normal(
        size=(2, 3, 10)
    ).astype(np.float32)

    torch_conv = torch.nn.Conv1d(
        in_channels=3,
        out_channels=4,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=True,
    )

    mytorch_conv = Conv1d(
        in_channels=3,
        out_channels=4,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=True,
    )

    run_conv_test(
        mytorch_conv,
        torch_conv,
        x,
    )


def test_conv1d_stride():
    rng = np.random.default_rng(1)

    x = rng.normal(
        size=(2, 3, 15)
    ).astype(np.float32)

    torch_conv = torch.nn.Conv1d(
        3,
        4,
        kernel_size=3,
        stride=2,
        padding=1,
    )

    mytorch_conv = Conv1d(
        3,
        4,
        kernel_size=3,
        stride=2,
        padding=1,
    )

    run_conv_test(
        mytorch_conv,
        torch_conv,
        x,
    )


def test_conv1d_dilation():
    rng = np.random.default_rng(2)

    x = rng.normal(
        size=(2, 3, 15)
    ).astype(np.float32)

    torch_conv = torch.nn.Conv1d(
        3,
        4,
        kernel_size=3,
        stride=1,
        padding=2,
        dilation=2,
    )

    mytorch_conv = Conv1d(
        3,
        4,
        kernel_size=3,
        stride=1,
        padding=2,
        dilation=2,
    )

    run_conv_test(
        mytorch_conv,
        torch_conv,
        x,
    )


# ============================================================
# Conv2D
# ============================================================

def test_conv2d_basic():
    """
    Basic Conv2D test.
    """

    rng = np.random.default_rng(3)

    x = rng.normal(
        size=(2, 3, 8, 10)
    ).astype(np.float32)

    torch_conv = torch.nn.Conv2d(
        in_channels=3,
        out_channels=4,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=True,
    )

    mytorch_conv = Conv2d(
        in_channels=3,
        out_channels=4,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=True,
    )

    run_conv_test(
        mytorch_conv,
        torch_conv,
        x,
    )


def test_conv2d_non_square_kernel():
    """
    Important test.

    Makes sure ConvNd does not accidentally assume
    kernel_height == kernel_width.
    """

    rng = np.random.default_rng(4)

    x = rng.normal(
        size=(2, 3, 10, 12)
    ).astype(np.float32)

    torch_conv = torch.nn.Conv2d(
        3,
        4,
        kernel_size=(3, 5),
        stride=1,
        padding=(1, 2),
        dilation=1,
    )

    mytorch_conv = Conv2d(
        3,
        4,
        kernel_size=(3, 5),
        stride=1,
        padding=(1, 2),
        dilation=1,
    )

    run_conv_test(
        mytorch_conv,
        torch_conv,
        x,
    )


def test_conv2d_stride():
    rng = np.random.default_rng(5)

    x = rng.normal(
        size=(2, 3, 11, 13)
    ).astype(np.float32)

    torch_conv = torch.nn.Conv2d(
        3,
        5,
        kernel_size=(3, 5),
        stride=(2, 3),
        padding=(1, 2),
    )

    mytorch_conv = Conv2d(
        3,
        5,
        kernel_size=(3, 5),
        stride=(2, 3),
        padding=(1, 2),
    )

    run_conv_test(
        mytorch_conv,
        torch_conv,
        x,
    )


def test_conv2d_dilation():
    """
    Particularly important for your implementation because
    col2im needs to correctly handle:

        kernel_position * dilation
    """

    rng = np.random.default_rng(6)

    x = rng.normal(
        size=(2, 3, 15, 16)
    ).astype(np.float32)

    torch_conv = torch.nn.Conv2d(
        3,
        4,
        kernel_size=(3, 3),
        stride=1,
        padding=2,
        dilation=2,
    )

    mytorch_conv = Conv2d(
        3,
        4,
        kernel_size=(3, 3),
        stride=1,
        padding=2,
        dilation=2,
    )

    run_conv_test(
        mytorch_conv,
        torch_conv,
        x,
    )


def test_conv2d_stride_and_dilation():
    """
    Test stride and dilation simultaneously.
    """

    rng = np.random.default_rng(7)

    x = rng.normal(
        size=(2, 3, 20, 21)
    ).astype(np.float32)

    torch_conv = torch.nn.Conv2d(
        3,
        4,
        kernel_size=(3, 4),
        stride=(2, 3),
        padding=(2, 3),
        dilation=(2, 2),
    )

    mytorch_conv = Conv2d(
        3,
        4,
        kernel_size=(3, 4),
        stride=(2, 3),
        padding=(2, 3),
        dilation=(2, 2),
    )

    run_conv_test(
        mytorch_conv,
        torch_conv,
        x,
    )


def test_conv2d_no_bias():
    """
    Make sure bias=False works and backward does not try
    to access a nonexistent bias.
    """

    rng = np.random.default_rng(8)

    x = rng.normal(
        size=(2, 3, 8, 8)
    ).astype(np.float32)

    torch_conv = torch.nn.Conv2d(
        3,
        4,
        kernel_size=3,
        padding=1,
        bias=False,
    )

    mytorch_conv = Conv2d(
        3,
        4,
        kernel_size=3,
        padding=1,
        bias=False,
    )

    run_conv_test(
        mytorch_conv,
        torch_conv,
        x,
    )


# ============================================================
# Conv3D
# ============================================================

def test_conv3d_basic():
    """
    Basic Conv3D test.
    """

    rng = np.random.default_rng(9)

    x = rng.normal(
        size=(2, 2, 6, 7, 8)
    ).astype(np.float32)

    torch_conv = torch.nn.Conv3d(
        in_channels=2,
        out_channels=3,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=True,
    )

    mytorch_conv = Conv3d(
        in_channels=2,
        out_channels=3,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=True,
    )

    run_conv_test(
        mytorch_conv,
        torch_conv,
        x,
    )


def test_conv3d_non_cubic_kernel():
    """
    Makes sure Conv3D supports different kernel sizes
    in each spatial dimension.
    """

    rng = np.random.default_rng(10)

    x = rng.normal(
        size=(2, 2, 8, 9, 10)
    ).astype(np.float32)

    torch_conv = torch.nn.Conv3d(
        2,
        3,
        kernel_size=(3, 4, 5),
        stride=1,
        padding=(1, 2, 2),
    )

    mytorch_conv = Conv3d(
        2,
        3,
        kernel_size=(3, 4, 5),
        stride=1,
        padding=(1, 2, 2),
    )

    run_conv_test(
        mytorch_conv,
        torch_conv,
        x,
    )


def test_conv3d_stride():
    rng = np.random.default_rng(11)

    x = rng.normal(
        size=(2, 2, 8, 9, 10)
    ).astype(np.float32)

    torch_conv = torch.nn.Conv3d(
        2,
        3,
        kernel_size=(3, 3, 3),
        stride=(2, 2, 2),
        padding=1,
    )

    mytorch_conv = Conv3d(
        2,
        3,
        kernel_size=(3, 3, 3),
        stride=(2, 2, 2),
        padding=1,
    )

    run_conv_test(
        mytorch_conv,
        torch_conv,
        x,
    )


def test_conv3d_dilation():
    rng = np.random.default_rng(12)

    x = rng.normal(
        size=(2, 2, 10, 10, 10)
    ).astype(np.float32)

    torch_conv = torch.nn.Conv3d(
        2,
        3,
        kernel_size=3,
        stride=1,
        padding=2,
        dilation=2,
    )

    mytorch_conv = Conv3d(
        2,
        3,
        kernel_size=3,
        stride=1,
        padding=2,
        dilation=2,
    )

    run_conv_test(
        mytorch_conv,
        torch_conv,
        x,
    )


# ============================================================
# Parameter representation tests
# ============================================================

def test_conv2d_integer_and_tuple_parameters():
    """
    These two configurations should be identical:

        kernel_size=3
        stride=2
        padding=1

    and:

        kernel_size=(3, 3)
        stride=(2, 2)
        padding=(1, 1)
    """

    rng = np.random.default_rng(13)

    x = rng.normal(
        size=(2, 3, 10, 10)
    ).astype(np.float32)

    conv_int = Conv2d(
        3,
        4,
        kernel_size=3,
        stride=2,
        padding=1,
        dilation=1,
    )

    conv_tuple = Conv2d(
        3,
        4,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        dilation=(1, 1),
    )

    # Give both convolutions identical parameters.
    conv_tuple.weight.data[...] = (
        conv_int.weight.data
    )

    if conv_int.bias is not None:
        conv_tuple.bias.data[...] = (
            conv_int.bias.data
        )

    output_int = conv_int(
        Tensor(x.copy())
    )

    output_tuple = conv_tuple(
        Tensor(x.copy())
    )

    assert_array_equal(
        output_int.data,
        output_tuple.data,
        name="integer/tuple parameter output",
    )


# ============================================================
# Parameter shape tests
# ============================================================

def test_conv1d_parameter_shapes():
    conv = Conv1d(
        in_channels=3,
        out_channels=5,
        kernel_size=7,
    )

    assert conv.weight.shape == (
        5,
        3,
        7,
    )

    assert conv.bias.shape == (5,)


def test_conv2d_parameter_shapes():
    conv = Conv2d(
        in_channels=3,
        out_channels=5,
        kernel_size=(3, 5),
    )

    assert conv.weight.shape == (
        5,
        3,
        3,
        5,
    )

    assert conv.bias.shape == (5,)


def test_conv3d_parameter_shapes():
    conv = Conv3d(
        in_channels=3,
        out_channels=5,
        kernel_size=(3, 4, 5),
    )

    assert conv.weight.shape == (
        5,
        3,
        3,
        4,
        5,
    )

    assert conv.bias.shape == (5,)


# ============================================================
# Invalid arguments
# ============================================================

def test_conv2d_invalid_in_channels():
    with pytest.raises(ValueError):
        Conv2d(
            in_channels=0,
            out_channels=4,
            kernel_size=3,
        )


def test_conv2d_invalid_out_channels():
    with pytest.raises(ValueError):
        Conv2d(
            in_channels=3,
            out_channels=0,
            kernel_size=3,
        )