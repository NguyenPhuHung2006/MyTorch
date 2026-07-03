from ..autograd.context import Context
from ..tensor import Tensor
from ..autograd.node import Node
import numpy as np

class Function:
    @classmethod
    def apply(cls, *args):
        ctx = Context()

        out_data = cls.forward(ctx, *args)

        requires_grad = any(
            t.requires_grad
            for t in args
            if isinstance(t, Tensor)
        )

        node = None
        if requires_grad:
            node = Node(
                function=cls,
                ctx=ctx,
                parents=args,
            )

        return Tensor(
            out_data,
            requires_grad=requires_grad,
            grad_fn=node,
        )
    
    @staticmethod
    def forward(ctx: Context, *args):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        pass