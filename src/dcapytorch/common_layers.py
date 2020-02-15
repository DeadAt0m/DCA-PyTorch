import torch
from  .dca import _ADNNBaseCellWeak
from functools import reduce


class Flatten(_ADNNBaseCellWeak):
    r"""
        Flattens the input. Does not affect the batch size.
        
        This module have a weak compatibility with ADNN(See modules/dca/addn_cell.py for help)
    """
    def __init__(self):
        super(Flatten,self).__init__()
    
    def reverse(self, input):
        return input.view(self.input_shape)
        
    def _forward(self, input):
        return input.view(input.size(0), -1)
    

class Reshape(_ADNNBaseCellWeak):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape
    
    def _check_shape(self, input):
        return reduce(lambda x, y: x*y, input.size()[1:], 1) == reduce(lambda x, y: x*y, self.shape, 1)
    
    def reverse(self, input):
        return input.view(self.input_shape)
    
    def _forward(self, input):
        if not self._check_shape(input):
            raise ValueError('Total size of new tensor must be unchanged')
        return input.view(input.size(0),*self.shape)
    
    def extra_repr(self):
        return 'to_shape=({})'.format('x'.join(map(str,self.shape)))
