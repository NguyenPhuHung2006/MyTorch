from ..autograd.function import Function
from ..autograd.context import Context
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# credit: ChatGPT. I gave up

class ConvNd(Function):
    @staticmethod
    def _im2col(
        x,
        kernel_size,
        stride,
        padding,
        dilation,
    ):
        # ---------------------------------------------------------
        # Shapes
        #
        # x:
        #     (N, Cin, spatial...)
        #
        # weight:
        #     (Cout, Cin, kernel...)
        #
        # Example for Conv2d:
        #
        # x      = (N, Cin, H, W)
        # weight = (Cout, Cin, Kh, Kw)
        # ---------------------------------------------------------
        
        spatial_ndim = x.ndim - 2
        
        if len(stride) != spatial_ndim:
            raise ValueError(
                "stride must have one value per spatial dimension"
            )
        
        if len(padding) != spatial_ndim:
            raise ValueError(
                "padding must have one value per spatial dimension"
            )
        
        if len(dilation) != spatial_ndim:
            raise ValueError(
                "dilation must have one value per spatial dimension"
            )
        
        # ---------------------------------------------------------
        # Effective kernel size
        #
        # dilation=1:
        #
        #   kernel = 3
        #   effective = 3
        #
        # dilation=2:
        #
        #   kernel = 3
        #   effective = 5
        #
        #   x . x . x
        # ---------------------------------------------------------
        
        effective_kernel_size = tuple(
            d * (k - 1) + 1
            for k, d in zip(kernel_size, dilation)
        )
        
        # ---------------------------------------------------------
        # Padding
        #
        # x:
        # (N, Cin, spatial...)
        #
        # pad only the spatial dimensions.
        # ---------------------------------------------------------
        
        pad_width = (
            (0, 0),                     # batch
            (0, 0),                     # channels
            *(
                (p, p)
                for p in padding
            ),
        )
        
        x_padded = np.pad(
            x,
            pad_width=pad_width,
            mode="constant",
            constant_values=0,
        )
        
        # ---------------------------------------------------------
        # Extract sliding windows
        #
        # For Conv2d:
        #
        # x_padded:
        #     (N, Cin, H, W)
        #
        # windows:
        #     (N, Cin, H', W', Kh, Kw)
        # ---------------------------------------------------------
        
        spatial_axes = tuple(
            range(2, 2 + spatial_ndim)
        )
        
        windows = sliding_window_view(
            x_padded,
            window_shape=effective_kernel_size,
            axis=spatial_axes,
        )
        
        # ---------------------------------------------------------
        # Apply stride
        #
        # Example:
        #
        # stride = (2, 2)
        #
        # Keep:
        #
        # 0, 2, 4, 6, ...
        # ---------------------------------------------------------
        
        spatial_slices = tuple(
            slice(None, None, s)
            for s in stride
        )
        
        # ---------------------------------------------------------
        # Apply dilation
        #
        # Example:
        #
        # dilation = (2, 2)
        #
        # effective window:
        #
        # x . x
        # . . .
        # x . x
        #
        # ---------------------------------------------------------
        
        kernel_slices = tuple(
            slice(None, None, d)
            for d in dilation
        )
        
        windows = windows[
            (
                slice(None),           # N
                slice(None),           # Cin
                *spatial_slices,        # output positions
                *kernel_slices,         # kernel positions
            )
        ]
        
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
        
        ctx.save_for_backward(
            x,
            weight,
            x_cols,
        )
        
        return output
    
    @staticmethod
    def _col2im(
        x_cols,
        x,
        kernel_size,
        stride,
        padding, 
        dilation,
        output_spatial_shape,
    ):
        
        

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        stride = ctx.saved_data["stride"]
        padding = ctx.saved_data["padding"]
        dilation = ctx.saved_data["dilation"]
        output_spatial_shape = ctx.saved_data["output_spatial_shape"]
        
        x, weight, x_cols = ctx.saved_tensors
        
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
            x.shape,
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
        
        