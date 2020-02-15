import pytest
from dcapytorch import *  
import torch
from functools import reduce
prod = lambda l: reduce(lambda x, y: x * y, l, 1)
cmp = lambda l1,l2: not bool(list(set(l1) - set(l2)))

def test_flatten():
    layer = Flatten()
    test_size = 5
    n = 5 
    x = torch.randn(24,*[test_size for i in range(n)])
    x1 = layer(x)
    assert x1.size()[-1] == test_size ** n,'flatten layer is broken'
    print('Test 1 passed')
    _x = torch.randn(24,*[test_size-1 for i in range(n)])
    x1, x2 = layer(x, pre_activation=_x)
    assert  x1.size()[-1] == (test_size-1) ** n and x2.size()[-1] == test_size ** n, 'flatten layer does not handle pre_activation  properly'
    print('Test 2 passed')
    assert torch.sum(abs(x - layer.reverse(layer(x)))) < 1e-5, 'reverse function does not working'
    print('Test 3 passed')
    
def test_reshape():
    test_size = 5
    n = 5 
    x = torch.randn(24,prod([test_size for i in range(n)]))
    layer = Reshape(*[test_size for i in range(n)])
    x1 = layer(x)
    assert cmp(x1.size()[1:],[test_size for i in range(n)]),'reshape layer is broken'
    print('Test 1 passed')
    x1, x2 = layer(x, pre_activation=x)
    assert cmp(x1.size()[1:],[test_size for i in range(n)]) and cmp(x2.size()[1:],[test_size for i in range(n)]), 'reshape layer does not handle pre_activation  properly'
    print('Test 2 passed')
    assert torch.sum(abs(x - layer.reverse(layer(x)))) < 1e-5, 'reverse function does not working'
    print('Test 3 passed')
