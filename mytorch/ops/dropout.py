from ..autograd.function import Function
from ..autograd.context import Context
import numpy as np

class Dropout(Function):
    @staticmethod
    def forward(ctx: Context, 
                x: np.ndarray, 
                p: float, 
                training: bool, 
                mask_shape: tuple
            ):
        
        if not training or p == 0:
            ctx.saved_data["training"] = False
            return x
        
        ctx.saved_data["p"] = p
        ctx.saved_data["training"] = True
        
        rand = np.random.rand(*mask_shape)
        mask = rand >= p
        ctx.save_for_backward(mask)
        return x * mask / (1 - p)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        
        if not ctx.saved_data["training"]:
            return (grad_output,)
        
        mask, = ctx.saved_tensors
        p = ctx.saved_data["p"]
        
        grad_x = grad_output * mask / (1 - p)
        return (grad_x, )
        
        
        