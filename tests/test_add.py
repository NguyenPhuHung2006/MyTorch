from mytorch.tensor import Tensor

def test_add_forward():
    x = Tensor([1, 2, 3])
    y = Tensor([4, 5, 6])

    z = x + y

    assert (z.data == [5, 7, 9]).all()