class Node:
    def __init__(self, function, ctx, parents=()):
        self.function = function
        self.ctx = ctx
        self.parents = parents