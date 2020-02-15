import pytest
from dcapytorch import *  
import torch
from torch.nn.modules.utils import _single, _pair, _triple
import  numpy as np


def calculate_pad(S, stride, dilation, kernel_size):
    from math import ceil
    tmp = []
    for i in range(len(S)):
        tmp.append(ceil((S[i]*(stride[i] - 1) - stride[i] + 1 + dilation[i]*(kernel_size[i] - 1)) / 2 ))
    return tuple(tmp)


def test_activation_function_init():
    in_feat = 512
    out_feat = 256
    test_cell = ADNNLinearCell(in_feat, out_feat,act_fn=torch.nn.ReLU)
    test = torch.zeros(10) - 1
    assert torch.all(torch.eq(test_cell.act_fn(test),torch.zeros(10))),  'activation function works incorrectly'
    

def test_weight_init():
    in_feat = 512
    out_feat = 256
    test_cell = ADNNLinearCell(in_feat, out_feat, weight_init=torch.nn.init.constant_, 
                               init_params = {'val':3})
    assert not (test_cell.bias is None), 'weight is None seems to be reset_parameters does not work correctly'
    
    assert not (test_cell.weight is None), 'weight is None seems to be reset_parameters does not work correctly'
    
    assert torch.all(torch.eq(test_cell.weight - 3,torch.zeros(out_feat,in_feat))), 'something wrong during weight_initializer init or reset_parameters method'
    print('linear cell weigth init - ok')
    
    in_feat = 64
    out_feat = 32
    kernel_size = 10
    #standard config
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, weight_init=torch.nn.init.constant_, 
                               init_params = {'val':3})
    assert not (test_cell.weight is None), 'weight is None seems to be reset_parameters does not work correctly'
    
    assert torch.all(torch.eq(test_cell.weight - 3,torch.zeros(out_feat,in_feat, kernel_size))), 'something wrong during weight_initializer init or reset_parameters method'
    print('conv cell weigth init - ok')

def test_init_ADNNLinearCell():
    batch_size = 64
    in_feat = 256
    out_feat = 256
    threshold  = 1e-2
    x = torch.randn(batch_size, in_feat)

    test_cell = ADNNLinearCell(in_feat, out_feat,weight_init=torch.nn.init.orthogonal_)
    print('Case INFO: ',test_cell)
    assert (x - test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),test_cell.weight,None)).sum() < threshold, 'operators does not works correctly'
        

def test_init_ADNNConv1dCell_case1():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 10
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    #standard config
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
    
    
def test_init_ADNNConv1dCell_case2():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 10
    stride = 2
    padding = 0
    dilation = 1
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
    
    
def test_init_ADNNConv1dCell_case3():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 10
    stride = 4
    padding = 0
    dilation = 1
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
                        
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
def test_init_ADNNConv1dCell_case4():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 10
    stride = 1
    padding = 0
    dilation = 2
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
def test_init_ADNNConv1dCell_case5():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 10
    stride = 1
    padding = 0
    dilation = 4
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
def test_init_ADNNConv1dCell_case6():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 10
    stride = 2
    padding = 0
    dilation = 2
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConv1dCell_case7():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 10
    stride = 1
    padding = 0
    dilation = 1
    groups = 64
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConv1dCell_case8():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 10
    stride = 2
    padding = 0
    dilation = 1
    groups = 64
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConv1dCell_case9():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 10
    stride = 2
    padding = 0
    dilation = 2
    groups = 64
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
def test_init_ADNNConv1dCell_case10():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 10
    stride = 2
    dilation = 1
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
    
    
def test_init_ADNNConv1dCell_case11():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 10
    stride = 4
    dilation = 1
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
def test_init_ADNNConv1dCell_case12():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 10
    stride = 1
    dilation = 2
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              None, stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
def test_init_ADNNConv1dCell_case13():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 10
    stride = 1
    dilation = 4
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
def test_init_ADNNConv1dCell_case14():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 10
    stride = 2
    dilation = 2
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConv1dCell_case15():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 10
    stride = 1
    dilation = 1
    groups = 64
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConv1dCell_case16():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 10
    stride = 2
    dilation = 1
    groups = 64
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConv1dCell_case17():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 10
    stride = 2
    dilation = 2
    groups = 64
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConv1dCell_case18():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 9
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    #standard config
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
    
    
def test_init_ADNNConv1dCell_case19():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 9
    stride = 2
    padding = 0
    dilation = 1
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
    
    
def test_init_ADNNConv1dCell_case20():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 9
    stride = 4
    padding = 0
    dilation = 1
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
                        
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
def test_init_ADNNConv1dCell_case21():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 9
    stride = 1
    padding = 0
    dilation = 2
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
def test_init_ADNNConv1dCell_case22():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 9
    stride = 1
    padding = 0
    dilation = 4
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
def test_init_ADNNConv1dCell_case23():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 9
    stride = 2
    padding = 0
    dilation = 2
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConv1dCell_case24():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 9
    stride = 1
    padding = 0
    dilation = 1
    groups = 64
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConv1dCell_case25():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 9
    stride = 2
    padding = 0
    dilation = 1
    groups = 64
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConv1dCell_case26():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 9
    stride = 2
    padding = 0
    dilation = 2
    groups = 64
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
def test_init_ADNNConv1dCell_case27():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 9
    stride = 2
    dilation = 1
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
    
    
def test_init_ADNNConv1dCell_case28():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 9
    stride = 4
    dilation = 1
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
def test_init_ADNNConv1dCell_case29():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 9
    stride = 1
    dilation = 2
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              None, stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
def test_init_ADNNConv1dCell_case30():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 9
    stride = 1
    dilation = 4
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
def test_init_ADNNConv1dCell_case31():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 9
    stride = 2
    dilation = 2
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConv1dCell_case32():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 9
    stride = 1
    dilation = 1
    groups = 64
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConv1dCell_case33():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 9
    stride = 2
    dilation = 1
    groups = 64
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConv1dCell_case34():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 9
    stride = 2
    dilation = 2
    groups = 64
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
def test_init_ADNNConv2dCell_case1():
    batch_size = 64
    feat_size = 64
    in_feat = 64
    out_feat = 32
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    stride=_pair(stride)
    padding=_pair(padding)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)
    #standard config
    test_cell = ADNNConv2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
    
def test_init_ADNNConv2dCell_case2():
    batch_size = 64
    feat_size = 64
    in_feat = 64
    out_feat = 32
    kernel_size = 3
    stride = 2
    padding = 0
    dilation = 1
    groups = 1
    stride=_pair(stride)
    padding=_pair(padding)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)

    test_cell = ADNNConv2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
    
    
def test_init_ADNNConv2dCell_case3():
    batch_size = 64
    feat_size = 64
    in_feat = 64
    out_feat = 32
    kernel_size = 3
    stride = 4
    padding = 0
    dilation = 1
    groups = 1
    stride=_pair(stride)
    padding=_pair(padding)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)

    test_cell = ADNNConv2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
def test_init_ADNNConv2dCell_case4():
    batch_size = 64
    feat_size = 64
    in_feat = 64
    out_feat = 32
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 2
    groups = 1
    stride=_pair(stride)
    padding=_pair(padding)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)

    test_cell = ADNNConv2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
def test_init_ADNNConv2dCell_case5():
    batch_size = 64
    feat_size = 64
    in_feat = 64
    out_feat = 32
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 4
    groups = 1
    stride=_pair(stride)
    padding=_pair(padding)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)

    test_cell = ADNNConv2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
def test_init_ADNNConv2dCell_case6():
    batch_size = 64
    feat_size = 64
    in_feat = 64
    out_feat = 32
    kernel_size = 3
    stride = 2
    padding = 0
    dilation = 2
    groups = 1
    stride=_pair(stride)
    padding=_pair(padding)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)

    test_cell = ADNNConv2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConv2dCell_case7():
    batch_size = 64
    feat_size = 64
    in_feat = 64
    out_feat = 64
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1
    groups = 64
    stride=_pair(stride)
    padding=_pair(padding)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)

    test_cell = ADNNConv2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConv2dCell_case8():
    batch_size = 64
    feat_size = 64
    in_feat = 64
    out_feat = 64
    kernel_size = 3
    stride = 2
    padding = 0
    dilation = 1
    groups = 64
    stride=_pair(stride)
    padding=_pair(padding)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)
    
    test_cell = ADNNConv2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConv2dCell_case9():
    batch_size = 64
    feat_size = 64
    in_feat = 64
    out_feat = 64
    kernel_size = 3
    stride = 2
    padding = 0
    dilation = 2
    groups = 64
    stride=_pair(stride)
    padding=_pair(padding)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)

    test_cell = ADNNConv2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
        

    
def test_init_ADNNConv2dCell_case10():
    batch_size = 64
    feat_size = 64
    in_feat = 64
    out_feat = 32
    kernel_size = 3
    stride = 2
    dilation = 1
    groups = 1
    stride=_pair(stride)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
    
def test_init_ADNNConv2dCell_case11():
    batch_size = 64
    feat_size = 64
    in_feat = 64
    out_feat = 32
    kernel_size = 3
    stride = 4
    dilation = 1
    groups = 1
    stride=_pair(stride)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
def test_init_ADNNConv2dCell_case12():
    batch_size = 64
    feat_size = 64
    in_feat = 64
    out_feat = 32
    kernel_size = 3
    stride = 1
    dilation = 2
    groups = 1
    stride=_pair(stride)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
def test_init_ADNNConv2dCell_case13():
    batch_size = 64
    feat_size = 64
    in_feat = 64
    out_feat = 32
    kernel_size = 3
    stride = 1
    dilation = 4
    groups = 1
    stride=_pair(stride)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
def test_init_ADNNConv2dCell_case14():
    batch_size = 64
    feat_size = 64
    in_feat = 64
    out_feat = 32
    kernel_size = 3
    stride = 2
    dilation = 2
    groups = 1
    stride=_pair(stride)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConv2dCell_case15():
    batch_size = 64
    feat_size = 64
    in_feat = 64
    out_feat = 64
    kernel_size = 3
    stride = 1
    dilation = 1
    groups = 64
    stride=_pair(stride)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConv2dCell_case16():
    batch_size = 64
    feat_size = 64
    in_feat = 64
    out_feat = 64
    kernel_size = 3
    stride = 2
    dilation = 1
    groups = 64
    stride=_pair(stride)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConv2dCell_case17():
    batch_size = 64
    feat_size = 64
    in_feat = 64
    out_feat = 64
    kernel_size = 3
    stride = 2
    dilation = 2
    groups = 64
    stride=_pair(stride)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'


#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
def test_init_ADNNConv3dCell_case1():
    batch_size = 64
    feat_size = 64
    in_feat = 6
    out_feat = 3
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    stride=_triple(stride)
    padding=_triple(padding)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)
    #standard config
    test_cell = ADNNConv3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
    
def test_init_ADNNConv3dCell_case2():
    batch_size = 64
    feat_size = 64
    in_feat = 6
    out_feat = 3
    kernel_size = 3
    stride = 2
    padding = 0
    dilation = 1
    groups = 1
    stride=_triple(stride)
    padding=_triple(padding)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)

    test_cell = ADNNConv3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
    
    
def test_init_ADNNConv3dCell_case3():
    batch_size = 64
    feat_size = 64
    in_feat = 6
    out_feat = 3
    kernel_size = 3
    stride = 4
    padding = 0
    dilation = 1
    groups = 1
    stride=_triple(stride)
    padding=_triple(padding)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)

    test_cell = ADNNConv3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
def test_init_ADNNConv3dCell_case4():
    batch_size = 64
    feat_size = 64
    in_feat = 6
    out_feat = 3
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 2
    groups = 1
    stride=_triple(stride)
    padding=_triple(padding)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)

    test_cell = ADNNConv3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
def test_init_ADNNConv3dCell_case5():
    batch_size = 64
    feat_size = 64
    in_feat = 6
    out_feat = 3
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 4
    groups = 1
    stride=_triple(stride)
    padding=_triple(padding)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)

    test_cell = ADNNConv3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
def test_init_ADNNConv3dCell_case6():
    batch_size = 64
    feat_size = 64
    in_feat = 6
    out_feat = 3
    kernel_size = 3
    stride = 2
    padding = 0
    dilation = 2
    groups = 1
    stride=_triple(stride)
    padding=_triple(padding)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)

    test_cell = ADNNConv3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConv3dCell_case7():
    batch_size = 64
    feat_size = 64
    in_feat = 3
    out_feat = 3
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1
    groups = 3
    stride=_triple(stride)
    padding=_triple(padding)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)
    test_cell = ADNNConv3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConv3dCell_case8():
    batch_size = 64
    feat_size = 64
    in_feat = 3
    out_feat = 3
    kernel_size = 3
    stride = 2
    padding = 0
    dilation = 1
    groups = 3
    stride=_triple(stride)
    padding=_triple(padding)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)
    
    test_cell = ADNNConv3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConv3dCell_case9():
    batch_size = 64
    feat_size = 64
    in_feat = 3
    out_feat = 3
    kernel_size = 3
    stride = 2
    padding = 0
    dilation = 2
    groups = 3
    stride=_triple(stride)
    padding=_triple(padding)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)
    test_cell = ADNNConv3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
        

    
def test_init_ADNNConv3dCell_case10():
    batch_size = 64
    feat_size = 64
    in_feat = 6
    out_feat = 3
    kernel_size = 3
    stride = 2
    dilation = 1
    groups = 1
    stride=_triple(stride)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
    
    
def test_init_ADNNConv3dCell_case11():
    batch_size = 64
    feat_size = 64
    in_feat = 6
    out_feat = 3
    kernel_size = 3
    stride = 4
    dilation = 1
    groups = 1
    stride=_triple(stride)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
def test_init_ADNNConv3dCell_case12():
    batch_size = 64
    feat_size = 64
    in_feat = 6
    out_feat = 3
    kernel_size = 3
    stride = 1
    dilation = 2
    groups = 1
    stride=_triple(stride)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
def test_init_ADNNConv3dCell_case13():
    batch_size = 64
    feat_size = 64
    in_feat = 6
    out_feat = 3
    kernel_size = 3
    stride = 1
    dilation = 4
    groups = 1
    stride=_triple(stride)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
def test_init_ADNNConv3dCell_case14():
    batch_size = 64
    feat_size = 64
    in_feat = 6
    out_feat = 3
    kernel_size = 3
    stride = 2
    dilation = 2
    groups = 1
    stride=_triple(stride)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConv3dCell_case15():
    batch_size = 64
    feat_size = 64
    in_feat = 3
    out_feat = 3
    kernel_size = 3
    stride = 1
    dilation = 1
    groups = 3
    stride=_triple(stride)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConv3dCell_case16():
    batch_size = 64
    feat_size = 64
    in_feat = 3
    out_feat = 3
    kernel_size = 3
    stride = 2
    dilation = 1
    groups = 3
    stride=_triple(stride)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConv3dCell_case17():
    batch_size = 64
    feat_size = 64
    in_feat = 3
    out_feat = 3
    kernel_size = 3
    stride = 2
    dilation = 2
    groups = 3
    stride=_triple(stride)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    print('Output padding: ', test_cell._output_padding(test_cell._operator(x,test_cell.weight,None),
                                              x.size(), stride=stride,
                                              padding=padding,
                                              kernel_size=kernel_size, dilation=dilation))
    output_size = np.array(test_cell._inverse_operator(test_cell._operator(x,test_cell.weight,None),
                                                       test_cell.weight,x.size()).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------    

def test_init_ADNNConvTranspose1dCell_case1():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 10
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    #standard config
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

def test_init_ADNNConvTranspose1dCell_case2():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 10
    stride = 2
    padding = 0
    dilation = 1
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
    
    
def test_init_ADNNConvTranspose1dCell_case3():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 10
    stride = 4
    padding = 0
    dilation = 1
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
                        
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
def test_init_ADNNConvTranspose1dCell_case4():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 10
    stride = 1
    padding = 0
    dilation = 2
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
def test_init_ADNNConvTranspose1dCell_case5():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 10
    stride = 1
    padding = 0
    dilation = 4
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
def test_init_ADNNConvTranspose1dCell_case6():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 10
    stride = 2
    padding = 0
    dilation = 2
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConvTranspose1dCell_case7():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 10
    stride = 1
    padding = 0
    dilation = 1
    groups = 64
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConvTranspose1dCell_case8():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 10
    stride = 2
    padding = 0
    dilation = 1
    groups = 64
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConvTranspose1dCell_case9():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 10
    stride = 2
    padding = 0
    dilation = 2
    groups = 64
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
        

    
def test_init_ADNNConvTranspose1dCell_case10():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 10
    stride = 2
    dilation = 1
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
    
    
def test_init_ADNNConvTranspose1dCell_case11():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 10
    stride = 4
    dilation = 1
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
def test_init_ADNNConvTranspose1dCell_case12():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 10
    stride = 1
    dilation = 2
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
def test_init_ADNNConvTranspose1dCell_case13():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 10
    stride = 1
    dilation = 4
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
def test_init_ADNNConvTranspose1dCell_case14():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 10
    stride = 2
    dilation = 2
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConvTranspose1dCell_case15():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 10
    stride = 1
    dilation = 1
    groups = 64
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)   
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Input size: {}. Output size: {}'.format(x.size()[2:],output_size[2:]))
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConvTranspose1dCell_case16():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 10
    stride = 2
    dilation = 1
    groups = 64
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConvTranspose1dCell_case17():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 10
    stride = 2
    dilation = 2
    groups = 64
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
    
def test_init_ADNNConvTranspose1dCell_case18():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 9
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    #standard config
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

def test_init_ADNNConvTranspose1dCell_case19():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 9
    stride = 2
    padding = 0
    dilation = 1
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
    
    
def test_init_ADNNConvTranspose1dCell_case20():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 9
    stride = 4
    padding = 0
    dilation = 1
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
                        
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
def test_init_ADNNConvTranspose1dCell_case21():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 9
    stride = 1
    padding = 0
    dilation = 2
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
def test_init_ADNNConvTranspose1dCell_case22():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 9
    stride = 1
    padding = 0
    dilation = 4
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
def test_init_ADNNConvTranspose1dCell_case23():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 9
    stride = 2
    padding = 0
    dilation = 2
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConvTranspose1dCell_case24():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 9
    stride = 1
    padding = 0
    dilation = 1
    groups = 64
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConvTranspose1dCell_case25():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 9
    stride = 2
    padding = 0
    dilation = 1
    groups = 64
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConvTranspose1dCell_case26():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 9
    stride = 2
    padding = 0
    dilation = 2
    groups = 64
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
        

    
def test_init_ADNNConvTranspose1dCell_case27():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 9
    stride = 2
    dilation = 1
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
    
    
def test_init_ADNNConvTranspose1dCell_case28():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 9
    stride = 4
    dilation = 1
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
def test_init_ADNNConvTranspose1dCell_case29():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 9
    stride = 1
    dilation = 2
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
def test_init_ADNNConvTranspose1dCell_case30():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 9
    stride = 1
    dilation = 4
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
def test_init_ADNNConvTranspose1dCell_case31():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 9
    stride = 2
    dilation = 2
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConvTranspose1dCell_case32():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 9
    stride = 1
    dilation = 1
    groups = 64
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConvTranspose1dCell_case33():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 9
    stride = 2
    dilation = 1
    groups = 64
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConvTranspose1dCell_case34():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 9
    stride = 2
    dilation = 2
    groups = 64
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
    
    
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
def test_init_ADNNConvTranspose2dCell_case1():
    batch_size = 64
    feat_size = 64
    in_feat = 32
    out_feat = 64
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    stride=_pair(stride)
    padding=_pair(padding)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)
    #standard config
    test_cell = ADNNConvTranspose2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
def test_init_ADNNConvTranspose2dCell_case2():
    batch_size = 64
    feat_size = 64
    in_feat = 32
    out_feat = 64
    kernel_size = 3
    stride = 2
    padding = 0
    dilation = 1
    groups = 1
    stride=_pair(stride)
    padding=_pair(padding)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)

    test_cell = ADNNConvTranspose2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
    
def test_init_ADNNConvTranspose2dCell_case3():
    batch_size = 64
    feat_size = 64
    in_feat = 32
    out_feat = 64
    kernel_size = 3
    stride = 4
    padding = 0
    dilation = 1
    groups = 1
    stride=_pair(stride)
    padding=_pair(padding)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)

    test_cell = ADNNConvTranspose2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'


    
    
def test_init_ADNNConvTranspose2dCell_case4():
    batch_size = 64
    feat_size = 64
    in_feat = 32
    out_feat = 64
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 2
    groups = 1
    stride=_pair(stride)
    padding=_pair(padding)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)

    test_cell = ADNNConvTranspose2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'


    
def test_init_ADNNConvTranspose2dCell_case5():
    batch_size = 64
    feat_size = 64
    in_feat = 32
    out_feat = 64
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 4
    groups = 1
    stride=_pair(stride)
    padding=_pair(padding)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'


    
    
def test_init_ADNNConvTranspose2dCell_case6():
    batch_size = 64
    feat_size = 64
    in_feat = 64
    out_feat = 32
    kernel_size = 3
    stride = 2
    padding = 0
    dilation = 2
    groups = 1
    stride=_pair(stride)
    padding=_pair(padding)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)

    test_cell = ADNNConvTranspose2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
def test_init_ADNNConvTranspose2dCell_case7():
    batch_size = 64
    feat_size = 64
    in_feat = 64
    out_feat = 64
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1
    groups = 64
    stride=_pair(stride)
    padding=_pair(padding)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)

    test_cell = ADNNConvTranspose2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
def test_init_ADNNConvTranspose2dCell_case8():
    batch_size = 64
    feat_size = 64
    in_feat = 64
    out_feat = 64
    kernel_size = 3
    stride = 2
    padding = 0
    dilation = 1
    groups = 64
    stride=_pair(stride)
    padding=_pair(padding)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)
    
    test_cell = ADNNConvTranspose2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
def test_init_ADNNConvTranspose2dCell_case9():
    batch_size = 64
    feat_size = 64
    in_feat = 64
    out_feat = 64
    kernel_size = 3
    stride = 2
    padding = 0
    dilation = 2
    groups = 64
    stride=_pair(stride)
    padding=_pair(padding)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)

    test_cell = ADNNConvTranspose2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

        

    
def test_init_ADNNConvTranspose2dCell_case10():
    batch_size = 64
    feat_size = 64
    in_feat = 32
    out_feat = 64
    kernel_size = 3
    stride = 2
    dilation = 1
    groups = 1
    stride=_pair(stride)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
    
    
def test_init_ADNNConvTranspose2dCell_case11():
    batch_size = 64
    feat_size = 64
    in_feat = 32
    out_feat = 64
    kernel_size = 3
    stride = 4
    dilation = 1
    groups = 1
    stride=_pair(stride)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
def test_init_ADNNConvTranspose2dCell_case12():
    batch_size = 64
    feat_size = 64
    in_feat = 32
    out_feat = 64
    kernel_size = 3
    stride = 1
    dilation = 2
    groups = 1
    stride=_pair(stride)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
def test_init_ADNNConvTranspose2dCell_case13():
    batch_size = 64
    feat_size = 64
    in_feat = 32
    out_feat = 64
    kernel_size = 3
    stride = 1
    dilation = 4
    groups = 1
    stride=_pair(stride)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
def test_init_ADNNConvTranspose2dCell_case14():
    batch_size = 64
    feat_size = 64
    in_feat = 32
    out_feat = 64
    kernel_size = 3
    stride = 2
    dilation = 2
    groups = 1
    stride=_pair(stride)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConvTranspose2dCell_case15():
    batch_size = 64
    feat_size = 64
    in_feat = 64
    out_feat = 64
    kernel_size = 3
    stride = 1
    dilation = 1
    groups = 64
    stride=_pair(stride)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConvTranspose2dCell_case16():
    batch_size = 64
    feat_size = 64
    in_feat = 64
    out_feat = 64
    kernel_size = 3
    stride = 2
    dilation = 1
    groups = 64
    stride=_pair(stride)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConvTranspose2dCell_case17():
    batch_size = 64
    feat_size = 64
    in_feat = 64
    out_feat = 64
    kernel_size = 3
    stride = 2
    dilation = 2
    groups = 64
    stride=_pair(stride)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
def test_init_ADNNConvTranspose3dCell_case1():
    batch_size = 64
    feat_size = 64
    in_feat = 3
    out_feat = 6
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    stride=_triple(stride)
    padding=_triple(padding)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)
    #standard config
    test_cell = ADNNConvTranspose3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
def test_init_ADNNConvTranspose3dCell_case2():
    batch_size = 64
    feat_size = 64
    in_feat = 3
    out_feat = 6
    kernel_size = 3
    stride = 2
    padding = 0
    dilation = 1
    groups = 1
    stride=_triple(stride)
    padding=_triple(padding)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)

    test_cell = ADNNConvTranspose3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
    
def test_init_ADNNConvTranspose3dCell_case3():
    batch_size = 64
    feat_size = 64
    in_feat = 3
    out_feat = 6
    kernel_size = 3
    stride = 4
    padding = 0
    dilation = 1
    groups = 1
    stride=_triple(stride)
    padding=_triple(padding)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)

    test_cell = ADNNConvTranspose3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'


    
    
def test_init_ADNNConvTranspose3dCell_case4():
    batch_size = 64
    feat_size = 64
    in_feat = 3
    out_feat = 6
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 2
    groups = 1
    stride=_triple(stride)
    padding=_triple(padding)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)

    test_cell = ADNNConvTranspose3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'


    
def test_init_ADNNConvTranspose3dCell_case5():
    batch_size = 64
    feat_size = 64
    in_feat = 3
    out_feat = 6
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 4
    groups = 1
    stride=_triple(stride)
    padding=_triple(padding)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)

    test_cell = ADNNConvTranspose3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'


    
    
def test_init_ADNNConvTranspose3dCell_case6():
    batch_size = 64
    feat_size = 64
    in_feat = 3
    out_feat = 6
    kernel_size = 3
    stride = 2
    padding = 0
    dilation = 2
    groups = 1
    stride=_triple(stride)
    padding=_triple(padding)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)

    test_cell = ADNNConvTranspose3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
def test_init_ADNNConvTranspose3dCell_case7():
    batch_size = 64
    feat_size = 64
    in_feat = 3
    out_feat = 3
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1
    groups = 3
    stride=_triple(stride)
    padding=_triple(padding)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)
    test_cell = ADNNConvTranspose3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
def test_init_ADNNConvTranspose3dCell_case8():
    batch_size = 64
    feat_size = 64
    in_feat = 3
    out_feat = 3
    kernel_size = 3
    stride = 2
    padding = 0
    dilation = 1
    groups = 3
    stride=_triple(stride)
    padding=_triple(padding)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)
    
    test_cell = ADNNConvTranspose3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
def test_init_ADNNConvTranspose3dCell_case9():
    batch_size = 64
    feat_size = 64
    in_feat = 3
    out_feat = 3
    kernel_size = 3
    stride = 2
    padding = 0
    dilation = 2
    groups = 3
    stride=_triple(stride)
    padding=_triple(padding)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)
    test_cell = ADNNConvTranspose3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,None)
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

        

    
def test_init_ADNNConvTranspose3dCell_case10():
    batch_size = 64
    feat_size = 64
    in_feat = 3
    out_feat = 6
    kernel_size = 3
    stride = 2
    dilation = 1
    groups = 1
    stride=_triple(stride)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
    
    
def test_init_ADNNConvTranspose3dCell_case11():
    batch_size = 64
    feat_size = 64
    in_feat = 3
    out_feat = 6
    kernel_size = 3
    stride = 4
    dilation = 1
    groups = 1
    stride=_triple(stride)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
def test_init_ADNNConvTranspose3dCell_case12():
    batch_size = 64
    feat_size = 64
    in_feat = 3
    out_feat = 6
    kernel_size = 3
    stride = 1
    dilation = 2
    groups = 1
    stride=_triple(stride)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
def test_init_ADNNConvTranspose3dCell_case13():
    batch_size = 64
    feat_size = 64
    in_feat = 3
    out_feat = 6
    kernel_size = 3
    stride = 1
    dilation = 4
    groups = 1
    stride=_triple(stride)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
def test_init_ADNNConvTranspose3dCell_case14():
    batch_size = 64
    feat_size = 64
    in_feat = 3
    out_feat = 6
    kernel_size = 3
    stride = 2
    dilation = 2
    groups = 1
    stride=_triple(stride)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConvTranspose3dCell_case15():
    batch_size = 64
    feat_size = 64
    in_feat = 3
    out_feat = 3
    kernel_size = 3
    stride = 1
    dilation = 1
    groups = 3
    stride=_triple(stride)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConvTranspose3dCell_case16():
    batch_size = 64
    feat_size = 64
    in_feat = 3
    out_feat = 3
    kernel_size = 3
    stride = 2
    dilation = 1
    groups = 3
    stride=_triple(stride)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'
    
def test_init_ADNNConvTranspose3dCell_case17():
    batch_size = 64
    feat_size = 64
    in_feat = 3
    out_feat = 3
    kernel_size = 3
    stride = 2
    dilation = 2
    groups = 3
    stride=_triple(stride)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    deconv = test_cell._operator(x,test_cell.weight,x.size())
    print('Input size: {}. Deconv size: {}'.format(x.size()[2:],deconv.size()[2:]))
    assert sum(np.array(x.size()[2:]) - np.array(deconv.size()[2:])) == 0, 'padding does not works correctly'
    output_size = np.array(test_cell._inverse_operator(deconv, test_cell.weight,None).size())
    print('Size differences: ',(np.array(x.size()) - output_size)[2:])
    assert sum(np.array(x.size()) - output_size) == 0, 'operators does not works correctly'

    
    
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
    
def test_cell_run_linear():
    batch_size = 64
    in_feat = 256
    out_feat = 256
    x = torch.randn(batch_size, in_feat)

    test_cell = ADNNLinearCell(in_feat, out_feat,act_fn=torch.nn.ReLU,weight_init=torch.nn.init.eye_)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    assert torch.all(torch.eq(x, first_run_preact)), 'operator works incorrect'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    

def test_cell_run_conv1d_case1():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 10
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    #standard config
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
def test_cell_run_conv1d_case2():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 10
    stride = 2
    padding = 0
    dilation = 1
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
    
def test_cell_run_conv1d_case3():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 10
    stride = 4
    padding = 0
    dilation = 1
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
                        
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
    
def test_cell_run_conv1d_case4():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 10
    stride = 1
    padding = 0
    dilation = 2
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'

    
def test_cell_run_conv1d_case5():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 10
    stride = 1
    padding = 0
    dilation = 4
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'

    
    
def test_cell_run_conv1d_case6():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 10
    stride = 2
    padding = 0
    dilation = 2
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
def test_cell_run_conv1d_case7():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 10
    stride = 1
    padding = 0
    dilation = 1
    groups = 64
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
def test_cell_run_conv1d_case8():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 10
    stride = 2
    padding = 0
    dilation = 1
    groups = 64
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
def test_cell_run_conv1d_case9():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 10
    stride = 2
    padding = 0
    dilation = 2
    groups = 64
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
def test_cell_run_conv1d_case10():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 10
    stride = 2
    dilation = 1
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
    
    
def test_cell_run_conv1d_case11():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 10
    stride = 4
    dilation = 1
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'

    
    
def test_cell_run_conv1d_case12():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 10
    stride = 1
    dilation = 2
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'

    
def test_cell_run_conv1d_case13():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 10
    stride = 1
    dilation = 4
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'

    
    
def test_cell_run_conv1d_case14():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 10
    stride = 2
    dilation = 2
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
def test_cell_run_conv1d_case15():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 10
    stride = 1
    dilation = 1
    groups = 64
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
def test_cell_run_conv1d_case16():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 10
    stride = 2
    dilation = 1
    groups = 64
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
def test_cell_run_conv1d_case17():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 10
    stride = 2
    dilation = 2
    groups = 64
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
def test_cell_run_conv1d_case18():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 9
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    #standard config
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
    
    
def test_cell_run_conv1d_case19():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 9
    stride = 2
    padding = 0
    dilation = 1
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
    
    
def test_cell_run_conv1d_case20():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 9
    stride = 4
    padding = 0
    dilation = 1
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
                        
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'

    
    
def test_cell_run_conv1d_case21():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 9
    stride = 1
    padding = 0
    dilation = 2
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
def test_cell_run_conv1d_case22():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 9
    stride = 1
    padding = 0
    dilation = 4
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'

    
    
def test_cell_run_conv1d_case23():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 9
    stride = 2
    padding = 0
    dilation = 2
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
def test_cell_run_conv1d_case24():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 9
    stride = 1
    padding = 0
    dilation = 1
    groups = 64
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
def test_cell_run_conv1d_case25():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 9
    stride = 2
    padding = 0
    dilation = 1
    groups = 64
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
def test_cell_run_conv1d_case26():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 9
    stride = 2
    padding = 0
    dilation = 2
    groups = 64
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'

    
def test_cell_run_conv1d_case27():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 9
    stride = 2
    dilation = 1
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
    
    
def test_cell_run_conv1d_case28():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 9
    stride = 4
    dilation = 1
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'

    
    
def test_cell_run_conv1d_case29():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 9
    stride = 1
    dilation = 2
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'

    
def test_cell_run_conv1d_case30():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 9
    stride = 1
    dilation = 4
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'

    
    
def test_cell_run_conv1d_case31():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 32
    kernel_size = 9
    stride = 2
    dilation = 2
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
def test_cell_run_conv1d_case32():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 9
    stride = 1
    dilation = 1
    groups = 64
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
def test_cell_run_conv1d_case33():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 9
    stride = 2
    dilation = 1
    groups = 64
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
def test_cell_run_conv1d_case34():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 9
    stride = 2
    dilation = 2
    groups = 64
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConv1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'

def test_cell_run_conv2d_case1():
    batch_size = 64
    feat_size = 64
    in_feat = 64
    out_feat = 32
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    stride=_pair(stride)
    padding=_pair(padding)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)
    #standard config
    test_cell = ADNNConv2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
def test_cell_run_conv3d_case1():
    batch_size = 64
    feat_size = 64
    in_feat = 6
    out_feat = 3
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    stride=_triple(stride)
    padding=_triple(padding)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)
    #standard config
    test_cell = ADNNConv3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
    
def test_cell_run_conv1dtranspose_case1():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 10
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    #standard config
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'

def test_cell_run_conv1dtranspose_case2():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 10
    stride = 2
    padding = 0
    dilation = 1
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
    
    
def test_cell_run_conv1dtranspose_case3():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 10
    stride = 4
    padding = 0
    dilation = 1
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
                        
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'

    
    
def test_cell_run_conv1dtranspose_case4():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 10
    stride = 1
    padding = 0
    dilation = 2
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'

    
def test_cell_run_conv1dtranspose_case5():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 10
    stride = 1
    padding = 0
    dilation = 4
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'

    
    
def test_cell_run_conv1dtranspose_case6():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 10
    stride = 2
    padding = 0
    dilation = 2
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
def test_cell_run_conv1dtranspose_case7():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 10
    stride = 1
    padding = 0
    dilation = 1
    groups = 64
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
def test_cell_run_conv1dtranspose_case8():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 10
    stride = 2
    padding = 0
    dilation = 1
    groups = 64
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
def test_cell_run_conv1dtranspose_case9():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 10
    stride = 2
    padding = 0
    dilation = 2
    groups = 64
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
        

    
def test_cell_run_conv1dtranspose_case10():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 10
    stride = 2
    dilation = 1
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
    
    
def test_cell_run_conv1dtranspose_case11():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 10
    stride = 4
    dilation = 1
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'

    
    
def test_cell_run_conv1dtranspose_case12():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 10
    stride = 1
    dilation = 2
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'

    
def test_cell_run_conv1dtranspose_case13():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 10
    stride = 1
    dilation = 4
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'

    
    
def test_cell_run_conv1dtranspose_case14():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 10
    stride = 2
    dilation = 2
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
def test_cell_run_conv1dtranspose_case15():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 10
    stride = 1
    dilation = 1
    groups = 64
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
def test_cell_run_conv1dtranspose_case16():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 10
    stride = 2
    dilation = 1
    groups = 64
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
def test_cell_run_conv1dtranspose_case17():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 10
    stride = 2
    dilation = 2
    groups = 64
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
    
def test_cell_run_conv1dtranspose_case18():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 9
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    #standard config
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'

def test_cell_run_conv1dtranspose_case19():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 9
    stride = 2
    padding = 0
    dilation = 1
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
    
def test_cell_run_conv1dtranspose_case20():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 9
    stride = 4
    padding = 0
    dilation = 1
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
                        
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'

    
    
def test_cell_run_conv1dtranspose_case21():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 9
    stride = 1
    padding = 0
    dilation = 2
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
def test_cell_run_conv1dtranspose_case22():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 9
    stride = 1
    padding = 0
    dilation = 4
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'

    
    
def test_cell_run_conv1dtranspose_case23():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 9
    stride = 2
    padding = 0
    dilation = 2
    groups = 1
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
def test_cell_run_conv1dtranspose_case24():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 9
    stride = 1
    padding = 0
    dilation = 1
    groups = 64
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
def test_cell_run_conv1dtranspose_case25():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 9
    stride = 2
    padding = 0
    dilation = 1
    groups = 64
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
def test_cell_run_conv1dtranspose_case26():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 9
    stride = 2
    padding = 0
    dilation = 2
    groups = 64
    stride=_single(stride)
    padding=_single(padding)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)

    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
        

    
def test_cell_run_conv1dtranspose_case27():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 9
    stride = 2
    dilation = 1
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
    
    
def test_cell_run_conv1dtranspose_case28():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 9
    stride = 4
    dilation = 1
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'

    
    
def test_cell_run_conv1dtranspose_case29():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 9
    stride = 1
    dilation = 2
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'

    
def test_cell_run_conv1dtranspose_case30():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 9
    stride = 1
    dilation = 4
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'

    
    
def test_cell_run_conv1dtranspose_case31():
    batch_size = 64
    feat_size = 100
    in_feat = 32
    out_feat = 64
    kernel_size = 9
    stride = 2
    dilation = 2
    groups = 1
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
def test_cell_run_conv1dtranspose_case32():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 9
    stride = 1
    dilation = 1
    groups = 64
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
def test_cell_run_conv1dtranspose_case33():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 9
    stride = 2
    dilation = 1
    groups = 64
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
def test_cell_run_conv1dtranspose_case34():
    batch_size = 64
    feat_size = 100
    in_feat = 64
    out_feat = 64
    kernel_size = 9
    stride = 2
    dilation = 2
    groups = 64
    stride=_single(stride)
    dilation=_single(dilation)
    kernel_size=_single(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size)
    padding = calculate_pad(tuple(x.size()[2:]), stride, dilation, kernel_size)
    test_cell = ADNNConvTranspose1dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'
    
def test_cell_run_conv12dtranspose_case1():
    batch_size = 64
    feat_size = 64
    in_feat = 32
    out_feat = 64
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    stride=_pair(stride)
    padding=_pair(padding)
    dilation=_pair(dilation)
    kernel_size=_pair(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size)
    #standard config
    test_cell = ADNNConvTranspose2dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'

def test_cell_run_conv3dtranspose_case1():
    batch_size = 64
    feat_size = 64
    in_feat = 3
    out_feat = 6
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    stride=_triple(stride)
    padding=_triple(padding)
    dilation=_triple(dilation)
    kernel_size=_triple(kernel_size)
    x = torch.randn(batch_size, in_feat, feat_size, feat_size,feat_size)
    #standard config
    test_cell = ADNNConvTranspose3dCell(in_feat, out_feat, kernel_size, 
                               stride=stride, padding=padding,
                               groups=groups,dilation=dilation)
    print('Case INFO: ',test_cell)
    unroll_step = 0
    first_run_act, first_run_preact = test_cell(x,x,x,x,unroll_step)
    assert first_run_act.size() ==  first_run_preact.size(), 'activation function change shape!'
    print('First forward run(feed-forward) passed')
    unroll_step = 1
    second_run_act, second_run_preact = test_cell(x,first_run_act, first_run_preact,x,unroll_step)
    assert second_run_act.size() ==  test_cell.lambdas.size(), 'activation size changed during adnn forward pass'
    print('Second forward run(adnn) passed')
    unroll_step = 2
    third_run_act, third_run_preact = test_cell(x,second_run_act, second_run_preact,x,unroll_step)
    print('Third forward run(adnn) passed')
    assert  not bool(torch.all(torch.eq(test_cell.lambdas,torch.zeros_like(first_run_act)))), 'something wrong with lambdas'

