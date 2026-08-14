from ..autograd.function import Function
from ..autograd.context import Context
import numpy as np
from ..utils.window import im2win, win2im

# credit: ChatGPT. I gave up

class ConvNd(Function):
    @staticmethod
    def _im2col(
        x: np.ndarray,
        kernel_size: tuple,
        stride: tuple,
        padding: tuple,
        dilation: tuple,
    ):
        spatial_ndim = len(kernel_size)
        
        windows = im2win(
            x, 
            kernel_size, 
            stride, 
            padding, 
            dilation
        )
        
        # ---------------------------------------------------------
        # Current shape:
        #
        # (N, Cin, O1, O2, ..., OD, K1, K2, ..., KD)
        #
        # We want:
        #
        # (N, O1, O2, ..., OD, Cin, K1, K2, ..., KD)
        #
        # This makes each spatial location correspond to one row
        # of our im2col matrix.
        # ---------------------------------------------------------
        
        windows = np.moveaxis(
            windows,
            1,
            1 + spatial_ndim,
        )
        
        # ---------------------------------------------------------
        # Extract output spatial shape
        # ---------------------------------------------------------
        
        output_spatial_shape = windows.shape[
            1 : 1 + spatial_ndim
        ]
        
        # ---------------------------------------------------------
        # im2col
        #
        # Convert:
        #
        # (N, O1, O2, ..., OD,
        #     Cin, K1, K2, ..., KD)
        #
        # into:
        #
        # (N * O1 * O2 * ... * OD,
        #     Cin * K1 * K2 * ... * KD)
        # ---------------------------------------------------------
        
        x_cols = windows.reshape(
            x.shape[0] * np.prod(output_spatial_shape),
            -1,
        )
        
        return x_cols, output_spatial_shape
        
    
    @staticmethod
    def forward(
        ctx: Context,
        x: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray | None,
        stride: tuple,
        padding: tuple,
        dilation: tuple,
    ):
        
        if weight.ndim != x.ndim:
            raise ValueError(
                "x and weight must have the same number of dimensions"
            )
        
        # ---------------------------------------------------------
        # Kernel size
        # ---------------------------------------------------------
        kernel_size = weight.shape[2:]
        
        x_cols, output_spatial_shape = ConvNd._im2col(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
        )
        
        # ---------------------------------------------------------
        # Flatten weights
        #
        # weight:
        #
        # (Cout, Cin, K1, K2, ..., KD)
        #
        # becomes:
        #
        # (Cout, Cin * K1 * K2 * ... * KD)
        # ---------------------------------------------------------
        
        weight_cols = weight.reshape(
            weight.shape[0],
            -1,
        )
        
        # ---------------------------------------------------------
        # Matrix multiplication
        #
        # cols:
        #     (N * output_size, kernel_size)
        #
        # weight_cols.T:
        #     (kernel_size, Cout)
        #
        # result:
        #     (N * output_size, Cout)
        # ---------------------------------------------------------
        
        output = x_cols @ weight_cols.T
        
        if bias is not None:
            # ---------------------------------------------------------
            # Add bias
            #
            # bias:
            #     (Cout,)
            #
            # NumPy broadcasts it across every output location.
            # ---------------------------------------------------------
            output += bias
        
        # ---------------------------------------------------------
        # Restore spatial dimensions
        #
        # Currently:
        #
        # (N * O1 * ... * OD, Cout)
        #
        # First:
        #
        # (N, O1, ..., OD, Cout)
        # ---------------------------------------------------------
        
        output = output.reshape(
            x.shape[0],
            *output_spatial_shape,
            weight.shape[0],
        )
        
        # ---------------------------------------------------------
        # Move Cout to channel dimension.
        #
        # (N, O1, ..., OD, Cout)
        #
        # ->
        #
        # (N, Cout, O1, ..., OD)
        # ---------------------------------------------------------
        
        output = np.moveaxis(
            output,
            -1,
            1,
        )
        
        ctx.saved_data["stride"] = stride
        ctx.saved_data["padding"] = padding
        ctx.saved_data["dilation"] = dilation
        ctx.saved_data["output_spatial_shape"] = output_spatial_shape
        ctx.saved_data["x_shape"] = x.shape
        
        ctx.save_for_backward(x_cols, weight)
        
        return output
    
    @staticmethod
    def _col2im(
        grad_x_cols: np.ndarray,
        x_shape: tuple,
        kernel_size: tuple,
        stride: tuple,
        padding: tuple,
        dilation: tuple,
        output_spatial_shape: tuple,
    ):
        spatial_ndim = len(kernel_size)
        N, C = x_shape[:2]
        
        # ---------------------------------------------------------
        # Reshape column gradients back into individual windows
        #
        # grad_x_cols:
        #
        #     (N * O1 * ... * OD,
        #      C * K1 * ... * KD)
        #
        # becomes:
        #
        #     (N,
        #      O1, ..., OD,
        #      C,
        #      K1, ..., KD)
        #
        # Each output position now has its corresponding
        # kernel-sized gradient window.
        # ---------------------------------------------------------
        
        windows = grad_x_cols.reshape(
            N,
            *output_spatial_shape,
            C,
            *kernel_size,
        )
        
        windows = np.moveaxis(
            windows,
            spatial_ndim + 1,
            1,
        )
        
        grad_x = win2im(
            windows,
            x_shape,
            kernel_size,
            stride,
            padding,
            dilation,
        )

        return grad_x
        

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        stride = ctx.saved_data["stride"]
        padding = ctx.saved_data["padding"]
        dilation = ctx.saved_data["dilation"]
        output_spatial_shape = ctx.saved_data["output_spatial_shape"]
        x_shape = ctx.saved_data["x_shape"]
        
        x_cols, weight = ctx.saved_tensors
        
        # --------------------------------------------------
        # Convert:
        #
        # (N, Cout, O1, ..., OD)
        #
        # ->
        #
        # (N * O1 * ... * OD, Cout)
        # --------------------------------------------------

        grad_output_cols = np.moveaxis(
            grad_output,
            1,
            -1,
        )

        grad_output_cols = grad_output_cols.reshape(
            -1,
            weight.shape[0],
        )

        # --------------------------------------------------
        # Weight gradient
        #
        # dW = dY.T @ X_col
        # --------------------------------------------------

        grad_weight = (
            grad_output_cols.T @ x_cols
        )

        grad_weight = grad_weight.reshape(
            weight.shape
        )

        # --------------------------------------------------
        # Bias gradient
        # --------------------------------------------------

        grad_bias = grad_output_cols.sum(
            axis=0
        )
        
        # --------------------------------------------------
        # Input gradient
        #
        # dX_col = dY @ W
        # --------------------------------------------------

        weight_cols = weight.reshape(
            weight.shape[0],
            -1,
        )

        grad_x_cols = (
            grad_output_cols @ weight_cols
        )
        
        # --------------------------------------------------
        # col2im
        # --------------------------------------------------

        grad_x = ConvNd._col2im(
            grad_x_cols,
            x_shape,
            weight.shape[2:],
            stride,
            padding,
            dilation,
            output_spatial_shape,
        )

        return (
            grad_x,
            grad_weight,
            grad_bias,
        )
        
        