from .context import Context
from ..tensor import Tensor
from .node import Node
import numpy as np

class Function:
    @classmethod
    def apply(cls, *args):
        ctx = Context()

        raw_args = [
            arg.data if isinstance(arg, Tensor) else arg
            for arg in args
        ]

        out_data = cls.forward(ctx, *raw_args)

        requires_grad = any(
            arg.requires_grad
            for arg in args
            if isinstance(arg, Tensor)
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