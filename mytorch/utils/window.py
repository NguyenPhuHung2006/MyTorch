import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def im2win(
    x: np.ndarray, 
    kernel_size: tuple,
    stride: tuple,
    padding: tuple,
    dilation: tuple,
    pad_value=0,
):
    # ---------------------------------------------------------
    # Shapes
    #
    # x:
    #     (N, Cin, spatial...)
    #
    # Example for Conv2d:
    #
    # x      = (N, Cin, H, W)
    # ---------------------------------------------------------
    
    spatial_ndim = len(kernel_size)
    
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
        constant_values=pad_value,
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
    
    return windows
    
def win2im(
    windows: np.ndarray,
    x_shape: tuple,
    kernel_size: tuple,
    stride: tuple,
    padding: tuple,
    dilation: tuple,
):
    spatial_ndim = len(kernel_size)
    N, C = x_shape[:2]
    input_spatial_shape = x_shape[2:]
    
    padded_input_spatial_shape = tuple(
        i + 2 * p
        for i, p in zip(
            input_spatial_shape,
            padding,
        )
    )
    
    effective_kernel_size = tuple(
        d * (k - 1) + 1
        for d, k in zip(
            dilation,
            kernel_size,
        )
    )
    
    prev_windows_spatial_shape = tuple(
        p_i - e_k + 1
        for p_i, e_k in zip(
            padded_input_spatial_shape,
            effective_kernel_size,
        )
    )
    
    prev_windows = np.zeros(
        (
            N,
            C,
            *prev_windows_spatial_shape,
            *effective_kernel_size,
        )
    )
    
    spatial_slice = tuple(
        slice(None, None, s)
        for s in stride
    )
    
    kernel_slice = tuple(
        slice(None, None, d)
        for d in dilation
    )
    
    slices = (slice(None), slice(None)) + spatial_slice + kernel_slice
    
    prev_windows[slices] = windows
    
    x_padded = np.zeros(
        (
            N,
            C,
            *padded_input_spatial_shape,
        )
    )
    
    for k_pos in np.ndindex(effective_kernel_size):
        x_slices = (
            (slice(None), slice(None))
            + tuple(
                slice(k_i, k_i + o_i)
                for k_i, o_i in zip(k_pos, prev_windows_spatial_shape)
            )
        )
    
        wnd_slices = (
            (slice(None), slice(None))
            + tuple(slice(None) for _ in range(spatial_ndim))
            + k_pos
        )
        
        x_padded[x_slices] += prev_windows[wnd_slices]
    
    unpad_slices = tuple(
        slice(
            p,
            p + size,
        )
        for p, size in zip(
            padding,
            input_spatial_shape,
        )
    )
    
    x = x_padded[
        (
            slice(None),
            slice(None),
            *unpad_slices,
        )
    ]
    
    assert x.shape == x_shape, f"expected x.shape to be {x_shape}, but got {x.shape}"
    
    return x