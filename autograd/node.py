from function import Function
from context import Context

class Node:
    def __init__(self, function: Function, ctx: Context, parents=()):
        self.function = function
        self.ctx = ctx
        self.parents = parents