def ntuple(n):
    def parse(value):
        if isinstance(value, int):
            return (value,) * n

        if not isinstance(value, tuple):
            raise TypeError(
                f"Expected an int or tuple of length {n}, "
                f"got {type(value).__name__}."
            )

        if len(value) != n:
            raise ValueError(
                f"Expected a tuple of length {n}, "
                f"got {len(value)}."
            )

        return value

    return parse