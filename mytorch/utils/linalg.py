import numpy as np

def transpose_last_two_dims(A):
    A = np.asarray(A)

    if A.ndim < 2:
        return A

    return np.swapaxes(A, -1, -2)