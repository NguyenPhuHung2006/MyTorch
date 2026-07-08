import numpy as np

def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]):
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
            
    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad

def transpose_last_two_dims(A):
    A = np.asarray(A)

    if A.ndim < 2:
        return A

    return np.swapaxes(A, -1, -2)
    