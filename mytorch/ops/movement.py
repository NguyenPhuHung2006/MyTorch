from ..autograd.function import Function
from ..autograd.context import Context
from ..tensor import Tensor
import numpy as np
from ..utils.broadcast import unbroadcast

class Transpose(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray, dim0: int, dim1: int):
        ctx.saved_data["dim0"] = dim0
        ctx.saved_data["dim1"] = dim1
        return np.swapaxes(x, dim0, dim1)
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        dim0 = ctx.saved_data["dim0"]
        dim1 = ctx.saved_data["dim1"]
        grad_x = np.swapaxes(grad_output, dim0, dim1)
        return (grad_x,)
    
class GetItem(Function):
    @staticmethod
    def unwrap_index(index):
        if isinstance(index, Tensor):
            return index.data
        if isinstance(index, tuple):
            return tuple(GetItem.unwrap_index(i) for i in index)
        if isinstance(index, list):
            return [GetItem.unwrap_index(i) for i in index]
        return index
    
    @staticmethod
    def forward(ctx: Context, x: np.ndarray, index):
        index = GetItem.unwrap_index(index)
        ctx.saved_data["index"] = index
        ctx.saved_data["x_shape"] = x.shape
        return x[index]
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        index = ctx.saved_data["index"]
        x_shape = ctx.saved_data["x_shape"]
        
        grad_x = np.zeros(x_shape, dtype=grad_output.dtype)
        np.add.at(grad_x, index, grad_output)
        
        return (grad_x,)
    
class Reshape(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray, shape: int | tuple):
        ctx.saved_data["original_shape"] = x.shape
        return x.reshape(shape)
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        original_shape = ctx.saved_data["original_shape"]
        grad_x = grad_output.reshape(original_shape)
        return (grad_x,)
    
class Clone(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray):
        return x.copy()
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        return (grad_output,)
        
class Cat(Function):
    @staticmethod
    def forward(ctx: Context, *tensors, axis):
        ctx.saved_data["axis"] = axis
        ctx.saved_data["sizes"] = [t.shape[axis] for t in tensors]
        return np.concatenate(tensors, axis=axis)
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        axis = ctx.saved_data["axis"]
        sizes = ctx.saved_data["sizes"]
        splits = np.cumsum(sizes[:-1])
        grads = np.split(
            grad_output,
            splits,
            axis=axis,
        )
        return tuple(grads)
    
class Stack(Function):
    @staticmethod
    def forward(ctx: Context, *tensors, axis):
        ctx.saved_data["axis"] = axis
        ctx.saved_data["num_tensors"] = len(tensors)
        return np.stack(tensors, axis=axis)
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        axis = ctx.saved_data["axis"]
        num_tensors = ctx.saved_data["num_tensors"]
        split_grads = np.split(
            grad_output,
            num_tensors,
            axis=axis,
        )
        
        grads = []
        for need, grad in zip(ctx.needs_input_grad, split_grads):
            grad = np.squeeze(grad, axis=axis)
            grads.append(grad if need else None)

        return tuple(grads)
        
def cat(tensors, axis=0):
    return Cat.apply(*tensors, axis=axis)

def stack(tensors, axis=0):
    return Stack.apply(*tensors, axis=axis)

class Expand(Function):
    def forward(ctx: Context, x: np.ndarray, shape):
        ctx.saved_data["input_shape"] = x.shape
        return np.broadcast_to(x, shape)
        
    def backward(ctx: Context, grad_output: np.ndarray):
        input_shape = ctx.saved_data["input_shape"]
        
        grad_x = unbroadcast(grad_output, input_shape)
        
        return (grad_x,)