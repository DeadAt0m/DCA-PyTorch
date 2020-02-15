import math
import torch
import inspect
import numpy as np
from itertools import chain
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
from torch.nn import Module
from torch.nn.modules.utils import _single, _pair, _triple


tup_sub_int = lambda x, i: tuple([j-i for j in x])



class _ADNNBaseCellWeak(Module):
    r"""
        Base module for all ADNN cells with WEAK compatibility.
        
        Note: adnn_compatibilty constant used for handling correct behaviour in ADNN module and cannot be changed
        in any derived modules! This prevent occasional errors during using random modules in ADNN
        Compatibilities types:
            - adnn_compatibility = 2 - Full. Module must handle all appropriate forward operation.
            - adnn_compatibility = 1 - Weak. Module cannot change tensor values during forward pass.

        _forward and reverse methods must be implemented in child module.
        Reverse method is inverse to _forward. For example, during _forward pass Flatten layer 
        saves shape of input tensor and reverse method can transform it back.
    """
    def __init__(self):
        super(_ADNNBaseCellWeak,self).__init__()
        self.adnn_compatibility = 1
    
    def reset_parameters(self):
        pass 
    
    def _forward(self, input):
        raise NotImplementedError
    
    def reverse(self, input):
        raise NotImplementedError
        
    def forward(self, input, **kwargs):
        self.input_shape = input.shape
        if 'pre_activation' in kwargs.keys() and not (kwargs['pre_activation'] is None):
            return self._forward(kwargs['pre_activation']), self._forward(input)
        return self._forward(input)



class _ADNNBaseCell(Module):
    r"""
        Base module for all ADNN cells and containing logic for alterating direction forward pass.
        
        Note: adnn_compatibilty constant used for handling correct behaviour in ADNN module and cannot be changed
        in any derived modules! This prevent occasional errors during using random modules in ADNN
        Compatibilities types:
            - adnn_compatibility = 2 - Full. Module must handle all appropriate forward operation
            - adnn_compatibility = 1 - Weak. Module cannot change tensor values during forward pass.
        
        
        Attributes: 
            output_shape(int) - Number of feature produced by the _operator (used only for intitializing bias weight).
            
            bias(bool) - Flag for including bias weight in computation. Note: follows the original article[1] bias is
                         substracted from pre-activation. However, this can be changed via bias_behaviour attribute.
                         Default: True.
                        
            bias_behaviour(str) - Can take following values ['-','+'], substruction or adding correspondingly.
                                  Default: '-' (substruction)
                                  
            rho(float) - Penality factor in augmented lagrangian. See details in [1].
                         Default: 0.5
                         
            act_fn(torch.nn.Module) - Activation function built above PyTorch Module.
                                      Default: None (linear activation)
                                      
            act_fn_param(dict) - Dictionary of parameters for activation function.
                                 Default: None
                                 
            weight_init(func) - Function for weight initialization, takes weight as input and change it inplace.
                                Additional arguments can be passed throught init_params attribute.
                                Default: None (He uniform)
                                
            init_params(dict) - Dictionary of additional parameters for weight initializtion function.
                                Default: None
                                
        Methods:
             Public:
                 reset_parameters - Reinitialize all weights.
                 
                 forward - Method realizing forward pass in PyTorch Module(see corresponding docs).
                           Arguments:
                               input(tensor) - Same is any PyTorch Module
                               unroll_step(int) - Setting to zero this behave as usual NN forward pass, 
                                             any arguments except input do not count, but must passed
                                             as dummy variables. If unroll_step > 0 it refers to 
                                             recurrent adnn forward pass(see details in [1])
                               pre_activation(tensor) - Pre-activation from previous step
                               activation(tensor) - Activation from previous step
                               prev_activation(tensor) - Activation from previous layer(or input if none layers behind)
                            Return:
                               pre_activation(tensor) - New pre-activation
                               activation(tensor) - New activation
                                    
                               
                  extra_repr - Verbosing additional parameters in child module. Must be implemnted!
             Private:
                 _operator - Apply defined operator to input. Must be implemnted!
                             Arguments:
                                x(tensor) - input.
                                weight(tensor) - weight should be applied to input by operator.
                                output_size(List[int]) - Target output shape if requied.
                             Return:
                                result(tensor)
                                
                 _inverse_operator - Apply inverse operator to  input. Must be implemnted! 
                                     Have same arguments as _operator.
                                     Take care about shape during implemnting:
                                     shape(input) == shape(_inverse_operator(_operator(input)))
                                     Return:
                                        result(tensor)
                
                 _standard_init - He uniform initialization for weight with a=sqrt(5).
                
                 _expand_bias - Expand bias dimension for consistency with pre-activation tensor.
                                Return:
                                    expanded_bias(tensor) having same shape as pre-activation
                
                 _adnn_forward - Apply one step of alternating direction forward pass.
                                 For more details see [1].
                                 Arguments:
                                    pre_activation(tensor) - Pre-activation from previous step
                                    activation(tensor) - Activation from previous step
                                    prev_activation(tensor) - Activation from previous layer(or input if none layers behind)
                                 Return:
                                    new_pre_activation(tensor)
                                    new_activation(tensor)
                
        References:
            [1] Murdock, C., Chang, M., & Lucey, S. (2018). Deep component analysis via alternating direction neural networks.
                In Proceedings of the European Conference on Computer Vision (ECCV) (pp. 820-836).   
    """
    
    
    __constants__ = ['bias','rho','adnn_compatibility']
    def __init__(self, output_shape, bias=True, bias_behaviour='-', rho=1, act_fn=None, act_fn_params=None, 
                 weight_init=None, init_params=None):
        super(_ADNNBaseCell, self).__init__()
        #set base module having full compatibility with adnn module
        #this means correct work of unrollment routine
        self.adnn_compatibility = 2
        if bias_behaviour in ['-','+']:
            self.bias_sign = 1 if bias_behaviour == '+' else -1
        else:
            raise ValueError('{} not in list of allowed values: {}'.format(bias_behaviour,'[ '+','.join(['-','+'])+' ]'))
        #No restrictions on activation function
        self._init_act_fn(act_fn,act_fn_params)
        self.rho = float(rho)
        self.register_parameter('weight', None)
        self.weight_initializer = None
        self.lambdas = None
        init_params = {} if init_params is None else init_params
        if not (weight_init is None):
            self.weight_initializer = lambda weight: weight_init(weight,**init_params)
        if bias:
            self.bias = Parameter(torch.Tensor(output_shape))
        else:
            self.register_parameter('bias', None)
       
    
    def _init_act_fn(self,act_fn,act_fn_params):
        act_fn_params = {} if act_fn_params is None else act_fn_params
        #check if activation function built using Module
        self.act_fn = lambda x:x
        if not (act_fn is None):
            if not inspect.isclass(act_fn):
                raise ValueError('act_fn is not a class instance')
            if torch.nn.modules.module.Module in inspect.getmro(act_fn):
                #init act_fn with parameters
                self.act_fn = act_fn(*act_fn_params)

    def reset_parameters(self):
        self.input_shape = None
        if self.weight_initializer:
            self.weight_initializer(self.weight)
        else:
            self._standard_init()
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        
    def extra_repr(self):
        raise NotImplementedError     
        
    def _operator(self, x, weight, output_size):
        raise NotImplementedError
        
    def _inverse_operator(self, x, weight, output_size):
        raise NotImplementedError
    
    def _standard_init(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        

    def forward(self, input, pre_activation=None, activation=None, prev_activation=None, unroll_step=0, old_activation=None):
        if self.input_shape is None:
            self.input_shape = input.shape
        if unroll_step == 0:       
            pre_activation = self._operator(input, self.weight, None)
            activation = self.act_fn(pre_activation + self.bias_sign * (0 if self.bias is None else self._expand_bias(self.bias)))
        else:
            if unroll_step == 1:
                self.lambdas = torch.zeros(activation.size(),device=activation.device)
            pre_activation,activation = self._adnn_forward(pre_activation, activation, prev_activation, old_activation)
        #handle common usual case, if all arguments beside inputs are using
        if (pre_activation is None) and (activation is None) and (prev_activation is None) and (not unroll_step):
            return activation
        return activation, pre_activation
    
    def _expand_bias(self,bias):
        return bias[[None,]]
    

    def _adnn_forward(self,pre_activation, activation, prev_activation, old_activation):
        self.lambdas = self.lambdas +  self.rho * (pre_activation - activation)
        tmp = activation - (1/self.rho) * self.lambdas    
        new_pre_activation = tmp +\
                                self._operator(prev_activation -\
                                self._inverse_operator(tmp,self.weight,prev_activation.shape),
                                               self.weight,tmp.shape) *  1/(1+self.rho)
        
        new_activation = new_pre_activation + (1/self.rho) * self.lambdas
        if old_activation is not None:
            new_activation = (1/(1+self.rho))*(old_activation + self.rho*new_activation)
        new_activation = self.act_fn(new_activation)
        return new_activation, new_pre_activation


     
class ADNNLinearCell(_ADNNBaseCell):
    r"""
        Linear ADNN cell applies linear transformation to the incoming data.
        Inherits from _ADNNBaseCell.
        
        Attributes:
             in_features(int) - Size of each input sample.
             
             out_features(int) -  Size of each output sample.
            
             bias(bool) - Flag for including bias weight in computation. Note: follows the original article[1] bias is
                          substracted from pre-activation. However, this can be changed via bias_behaviour attribute.
                          Default: True.
                        
             bias_behaviour(str) - Can take following values ['-','+'], substruction or adding correspondingly.
                                   Default: '-' (substruction)
                                  
             rho(float) - Penality factor in augmented lagrangian. See details in [1].
                          Default: 0.5
                         
             act_fn(torch.nn.Module) - Activation function built above PyTorch Module.
                                       Default: None (linear activation)
                                      
             act_fn_param(dict) - Dictionary of parameters for activation function.
                                  Default: None
                                 
             weight_init(func) - Function for weight initialization, takes weight as input and change it inplace.
                                 Additional arguments can be passed throught init_params attribute.
                                 Default: None (He uniform)
                                
             init_params(dict) - Dictionary of additional parameters for weight initializtion function.
                                 Default: None
    
        
    """
    def __init__(self, in_features, out_features, bias=True, bias_behaviour='-', rho=1.0,  act_fn=None, act_fn_params=None,
                 weight_init=None, init_params=None):
        super(ADNNLinearCell, self).__init__(out_features, bias, bias_behaviour,
                                             rho, act_fn, act_fn_params,
                                             weight_init, init_params)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        super(ADNNLinearCell, self).reset_parameters()
        
    def _operator(self, x, weight, output_size):
        return F.linear(x, weight)
    

    def _inverse_operator(self, x, weight,output_size):
        return F.linear(x, weight.transpose(0,1))
       
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features,
                                                                 self.out_features, False if self.bias is None else True)



class _ADNNConvNdCell(_ADNNBaseCell):
    r"""
        Base module for all adnn cells used convolution as operator.
        Inherits from _ADNNBaseCell.
        
        Attributes:
             in_channels(int) - Number of channels in the input tensor.
             
             out_channels(int) - Number of channels produced by the convolution.
             
             kernel_size(int) - Size of the convolving kernel.
             
             stride(int or tuple) - Stride of the convolution.
             
             padding (int or tuple) - Zero-padding added to both sides of the input.
             
             dilation (int or tuple) - Spacing between kernel elements.
             
             groups (int) - Number of blocked connections from input channels to output channels.
             
             transposed(bool) - Ancillary variable to control weight initialization for transposed convolutions.
             
             output_padding(int or tuple) - Additional size added to one side of the output shape. 
             
             padding_mode(str) - Used only for extra output padding in transposed convolution if kernel_size is even.
                                 Can take following values ['constant', 'reflect', 'replicate'].
                                 Default: 'constant'
                                 
            
             bias(bool) - Flag for including bias weight in computation. Note: follows the original article[1] bias is
                          substracted from pre-activation. However, this can be changed via bias_behaviour attribute.
                          Default: True.
                        
             bias_behaviour(str) - Can take following values ['-','+'], substruction or adding correspondingly.
                                   Default: '-' (substruction)
                                  
             rho(float) - Penality factor in augmented lagrangian. See details in [1].
                          Default: 0.5
                         
             act_fn(torch.nn.Module) - Activation function built above PyTorch Module.
                                       Default: None (linear activation)
                                      
             act_fn_param(dict) - Dictionary of parameters for activation function.
                                  Default: None
                                 
             weight_init(func) - Function for weight initialization, takes weight as input and change it inplace.
                                 Additional arguments can be passed throught init_params attribute.
                                 Default: None (He uniform)
                                
             init_params(dict) - Dictionary of additional parameters for weight initializtion function.
                                 Default: None
        Methods:
             Private:  
                 _output_padding - Calculate size of output padding for transposed convolution.
                                   Return:
                                       result(tuple[int]) 
                 
                 _expand_bias - Expand bias dimension for consistency with pre-activation tensor.
                                Return:
                                    expanded_bias(tensor) having same shape as pre-activation
                
                 _extra_padding - Handle cases when built-in output padding cannot pad to expected shape.
                                  Usualy it happens when input size values and kenrel size values are even.
                                  Return:
                                      result(tensor)
                
                 _inverse_extra_padding - Annul consequencies of _extra_padding.
                                          Return:
                                              result(tensor)
                                
    """
    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias', 'rho',
                     'kernel_size', 'dim_size', 'output_padding','padding_mode']        
    
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias, bias_behaviour='-',
                 rho=0.5, act_fn = None, act_fn_params=None, weight_init=None, init_params=None,
                 padding_mode='constant'):
        super(_ADNNConvNdCell, self).__init__(out_channels, bias, bias_behaviour, rho, act_fn, act_fn_params,
                                      weight_init, init_params)
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups  
        self.padding_mode = padding_mode
        self.extra_padding_flag = False
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        super(_ADNNConvNdCell, self).reset_parameters()
    
    

    def _output_padding(self, input, output_size, stride, padding, kernel_size, dilation):
        # type: (Tensor, Optional[List[int]], List[int], List[int], List[int]) -> List[int]
        if output_size is None:
            ret = _single(self.output_padding)  # converting to list if was not already
        else:
            output_size = torch.jit._unwrap_optional(output_size)
            k = input.dim() - 2
            if len(output_size) == k + 2:
                output_size = output_size[2:]
            if len(output_size) != k:
                raise ValueError(
                    "output_size must have {} or {} elements (got {})"
                    .format(k, k + 2, len(output_size)))

            min_sizes = []
            max_sizes = []
            for d in range(k):
                dim_size = ((input.size(d + 2) - 1) * stride[d] -
                            2 * padding[d] + dilation[d]*(kernel_size[d]-1) + 1)
                        
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)

            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    return _single(0)

            res = []
            for d in range(k):
                res.append(output_size[d] - min_sizes[d])

            ret = res
        return ret
    
    def _expand_bias(self,bias):
        return self.bias[[None,...]+[None,]*len(self.kernel_size)]
    
    def _extra_padding(self,x,output_size):
        if not (output_size is None):
            diff = np.array(output_size)[2:] - np.array(x.size())[2:]
            if sum(diff) != 0:
                self.extra_padding_diff = tuple(chain.from_iterable([[i//2+1, i//2] for i in diff]))
                self.extra_padding_flag = True
                return F.pad(x, self.extra_padding_diff, self.padding_mode, value=0)
        return x

    def _inverse_extra_padding(self,x):
        if self.extra_padding_flag:
            return F.pad(x, self.extra_padding_diff, self.padding_mode, value=0)
        return x
    
    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)        

    
    

class ADNNConv1dCell(_ADNNConvNdCell):
    r"""
        Applies a 1D convolution over an input signal composed of several input channels
        Inherits from _ADNNConvNdCell.
        
        Attributes:
             in_channels(int) - Number of channels in the input tensor.
             
             out_channels(int) - Number of channels produced by the convolution.
             
             kernel_size(int) - Size of the convolving kernel.
             
             stride(int or tuple) - Stride of the convolution.
                                    Default: 1
             
             padding (int or tuple) - Zero-padding added to both sides of the input.
                                      Default: 0
             
             dilation (int or tuple) - Spacing between kernel elements.
                                       Default: 1
             
             groups (int) - Number of blocked connections from input channels to output channels.
                            Default: 1
             
             
             padding_mode(str) - Used only for extra output padding in transposed convolution if kernel_size is even.
                                 Can take following values ['constant', 'reflect', 'replicate'].
                                 Default: 'constant'
                                 
            
             bias(bool) - Flag for including bias weight in computation. Note: follows the original article[1] bias is
                          substracted from pre-activation. However, this can be changed via bias_behaviour attribute.
                          Default: True.
                        
             bias_behaviour(str) - Can take following values ['-','+'], substruction or adding correspondingly.
                                   Default: '-' (substruction)
                                  
             rho(float) - Penality factor in augmented lagrangian. See details in [1].
                          Default: 0.5
                         
             act_fn(torch.nn.Module) - Activation function built above PyTorch Module.
                                       Default: None (linear activation)
                                      
             act_fn_param(dict) - Dictionary of parameters for activation function.
                                  Default: None
                                 
             weight_init(func) - Function for weight initialization, takes weight as input and change it inplace.
                                 Additional arguments can be passed throught init_params attribute.
                                 Default: None (He uniform)
                                
             init_params(dict) - Dictionary of additional parameters for weight initializtion function.
                                 Default: None
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,  bias_behaviour='-', rho=0.5,
                 act_fn=None, act_fn_params=None, weight_init=None, init_params=None):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(ADNNConv1dCell, self).__init__(in_channels, out_channels, kernel_size,
                                             stride, padding, dilation, False, _single(0),
                                             groups, bias, bias_behaviour, rho, act_fn, act_fn_params,
                                             weight_init, init_params)
    

    def _operator(self, x, weight, output_size):
        return F.conv1d(x, weight, None, self.stride,
                        self.padding, self.dilation, self.groups)


    def _inverse_operator(self, x, weight, output_size):
        output_padding = self._output_padding(x, output_size, self.stride,
                                              self.padding, self.kernel_size, self.dilation)
        return F.conv_transpose1d(x, weight, None, self.stride, self.padding,
                                  output_padding, self.groups, self.dilation)

        


class ADNNConv2dCell(_ADNNConvNdCell):
    """
        Applies a 2D convolution over an input signal composed of several input channels
        Inherits from _ADNNConvNdCell.
        
        Attributes:
             in_channels(int) - Number of channels in the input tensor.
             
             out_channels(int) - Number of channels produced by the convolution.
             
             kernel_size(int) - Size of the convolving kernel.
             
             stride(int or tuple) - Stride of the convolution.
                                    Default: 1
             
             padding (int or tuple) - Zero-padding added to both sides of the input.
                                      Default: 0
             
             dilation (int or tuple) - Spacing between kernel elements.
                                       Default: 1
             
             groups (int) - Number of blocked connections from input channels to output channels.
                            Default: 1
             
             
             padding_mode(str) - Used only for extra output padding in transposed convolution if kernel_size is even.
                                 Can take following values ['constant', 'reflect', 'replicate'].
                                 Default: 'constant'
                                 
            
             bias(bool) - Flag for including bias weight in computation. Note: follows the original article[1] bias is
                          substracted from pre-activation. However, this can be changed via bias_behaviour attribute.
                          Default: True.
                        
             bias_behaviour(str) - Can take following values ['-','+'], substruction or adding correspondingly.
                                   Default: '-' (substruction)
                                  
             rho(float) - Penality factor in augmented lagrangian. See details in [1].
                          Default: 0.5
                         
             act_fn(torch.nn.Module) - Activation function built above PyTorch Module.
                                       Default: None (linear activation)
                                      
             act_fn_param(dict) - Dictionary of parameters for activation function.
                                  Default: None
                                 
             weight_init(func) - Function for weight initialization, takes weight as input and change it inplace.
                                 Additional arguments can be passed throught init_params attribute.
                                 Default: None (He uniform)
                                
             init_params(dict) - Dictionary of additional parameters for weight initializtion function.
                                 Default: None
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, bias_behaviour='-', rho=0.5,
                 act_fn=None, act_fn_params=None, weight_init=None, init_params=None):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(ADNNConv2dCell, self).__init__(in_channels, out_channels, kernel_size,
                                             stride, padding, dilation, False, _pair(0),
                                             groups, bias, bias_behaviour, rho, act_fn, act_fn_params,
                                             weight_init, init_params)


    def _operator(self, x, weight, output_size):
        return F.conv2d(x, weight, None, self.stride,
                        self.padding, self.dilation, self.groups)


    def _inverse_operator(self, x, weight, output_size):
        output_padding = self._output_padding(x, output_size, self.stride,
                                              self.padding, self.kernel_size, self.dilation)
        return F.conv_transpose2d(x, weight, None, self.stride, self.padding,
                                  output_padding, self.groups, self.dilation)    




class ADNNConv3dCell(_ADNNConvNdCell):
    """
        Applies a 3D convolution over an input signal composed of several input channels
        Inherits from _ADNNConvNdCell.
        
        Attributes:
             in_channels(int) - Number of channels in the input tensor.
             
             out_channels(int) - Number of channels produced by the convolution.
             
             kernel_size(int) - Size of the convolving kernel.
             
             stride(int or tuple) - Stride of the convolution.
                                    Default: 1
             
             padding (int or tuple) - Zero-padding added to both sides of the input.
                                      Default: 0
             
             dilation (int or tuple) - Spacing between kernel elements.
                                       Default: 1
             
             groups (int) - Number of blocked connections from input channels to output channels.
                            Default: 1
             
             
             padding_mode(str) - Used only for extra output padding in transposed convolution if kernel_size is even.
                                 Can take following values ['constant', 'reflect', 'replicate'].
                                 Default: 'constant'
                                 
            
             bias(bool) - Flag for including bias weight in computation. Note: follows the original article[1] bias is
                          substracted from pre-activation. However, this can be changed via bias_behaviour attribute.
                          Default: True.
                        
             bias_behaviour(str) - Can take following values ['-','+'], substruction or adding correspondingly.
                                   Default: '-' (substruction)
                                  
             rho(float) - Penality factor in augmented lagrangian. See details in [1].
                          Default: 0.5
                         
             act_fn(torch.nn.Module) - Activation function built above PyTorch Module.
                                       Default: None (linear activation)
                                      
             act_fn_param(dict) - Dictionary of parameters for activation function.
                                  Default: None
                                 
             weight_init(func) - Function for weight initialization, takes weight as input and change it inplace.
                                 Additional arguments can be passed throught init_params attribute.
                                 Default: None (He uniform)
                                
             init_params(dict) - Dictionary of additional parameters for weight initializtion function.
                                 Default: None
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, bias_behaviour='-', rho=0.5,
                 act_fn=None, act_fn_params=None, weight_init=None, init_params=None):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super(ADNNConv3dCell, self).__init__(in_channels, out_channels, kernel_size,
                                             stride, padding, dilation, False, _triple(0),
                                             groups, bias, bias_behaviour, rho, act_fn, act_fn_params,
                                             weight_init, init_params)
        

    def _operator(self, x, weight, output_size):
        return F.conv3d(x, weight, None, self.stride,
                        self.padding, self.dilation, self.groups)


    def _inverse_operator(self, x, weight, output_size):
        output_padding = self._output_padding(x, output_size, self.stride,
                                              self.padding, self.kernel_size, self.dilation)
        return F.conv_transpose3d(x, weight, None, self.stride, self.padding,
                                  output_padding, self.groups, self.dilation)
    



class ADNNConvTranspose1dCell(_ADNNConvNdCell):
    """
        Applies a 1D transposed convolution(deconvolution) over an input signal composed of several input channels
        Inherits from _ADNNConvNdCell.
        
        Attributes:
             in_channels(int) - Number of channels in the input tensor.
             
             out_channels(int) - Number of channels produced by the convolution.
             
             kernel_size(int) - Size of the convolving kernel.
             
             stride(int or tuple) - Stride of the convolution.
                                    Default: 1
             
             padding (int or tuple) - Zero-padding added to both sides of the input.
                                      Default: 0
             
             output_padding(int or tuple) - Additional size added to one side of the output shape.
                                            Default: 0
             
             dilation (int or tuple) - Spacing between kernel elements.
                                       Default: 1
             
             groups (int) - Number of blocked connections from input channels to output channels.
                            Default: 1
             
             
             padding_mode(str) - Used only for extra output padding in transposed convolution if kernel_size is even.
                                 Can take following values ['constant', 'reflect', 'replicate'].
                                 Default: 'constant'
                                 
            
             bias(bool) - Flag for including bias weight in computation. Note: follows the original article[1] bias is
                          substracted from pre-activation. However, this can be changed via bias_behaviour attribute.
                          Default: True.
                        
             bias_behaviour(str) - Can take following values ['-','+'], substruction or adding correspondingly.
                                   Default: '-' (substruction)
                                  
             rho(float) - Penality factor in augmented lagrangian. See details in [1].
                          Default: 0.5
                         
             act_fn(torch.nn.Module) - Activation function built above PyTorch Module.
                                       Default: None (linear activation)
                                      
             act_fn_param(dict) - Dictionary of parameters for activation function.
                                  Default: None
                                 
             weight_init(func) - Function for weight initialization, takes weight as input and change it inplace.
                                 Additional arguments can be passed throught init_params attribute.
                                 Default: None (He uniform)
                                
             init_params(dict) - Dictionary of additional parameters for weight initializtion function.
                                 Default: None
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, bias_behaviour='-',
                 dilation=1, rho=0.5, act_fn=None, act_fn_params=None,
                 weight_init=None, init_params=None):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        output_padding = _single(output_padding)
        super(ADNNConvTranspose1dCell, self).__init__(in_channels, out_channels, kernel_size,
                                                      stride, padding, dilation, True, output_padding,
                                                      groups, bias, bias_behaviour, rho, act_fn, act_fn_params,
                                                      weight_init, init_params)


    def _operator(self, x, weight, output_size):
        # type: (Tensor, Optional[List[int]]) -> Tensor
        output_padding = self._output_padding(x, output_size, self.stride,
                                              self.padding, self.kernel_size, self.dilation)
        res =  F.conv_transpose1d(x, weight, None, self.stride, self.padding,
                                  output_padding, self.groups, self.dilation)
        return self._extra_padding(res,output_size)

 


    def _inverse_operator(self, x, weight, output_size):
        return self._inverse_extra_padding(F.conv1d(x, weight, None, self.stride,
                                                    tup_sub_int(self.padding,
                                                                bool(self.extra_padding_flag)),
                                                    self.dilation, self.groups))

        

class ADNNConvTranspose2dCell(_ADNNConvNdCell):
    """
        Applies a 2D transposed convolution(deconvolution) over an input signal composed of several input channels
        Inherits from _ADNNConvNdCell.
        
        Attributes:
             in_channels(int) - Number of channels in the input tensor.
             
             out_channels(int) - Number of channels produced by the convolution.
             
             kernel_size(int) - Size of the convolving kernel.
             
             stride(int or tuple) - Stride of the convolution.
                                    Default: 1
             
             padding (int or tuple) - Zero-padding added to both sides of the input.
                                      Default: 0
             
             output_padding(int or tuple) - Additional size added to one side of the output shape.
                                            Default: 0
             
             dilation (int or tuple) - Spacing between kernel elements.
                                       Default: 1
             
             groups (int) - Number of blocked connections from input channels to output channels.
                            Default: 1
             
             
             padding_mode(str) - Used only for extra output padding in transposed convolution if kernel_size is even.
                                 Can take following values ['constant', 'reflect', 'replicate'].
                                 Default: 'constant'
                                 
            
             bias(bool) - Flag for including bias weight in computation. Note: follows the original article[1] bias is
                          substracted from pre-activation. However, this can be changed via bias_behaviour attribute.
                          Default: True.
                        
             bias_behaviour(str) - Can take following values ['-','+'], substruction or adding correspondingly.
                                   Default: '-' (substruction)
                                  
             rho(float) - Penality factor in augmented lagrangian. See details in [1].
                          Default: 0.5
                         
             act_fn(torch.nn.Module) - Activation function built above PyTorch Module.
                                       Default: None (linear activation)
                                      
             act_fn_param(dict) - Dictionary of parameters for activation function.
                                  Default: None
                                 
             weight_init(func) - Function for weight initialization, takes weight as input and change it inplace.
                                 Additional arguments can be passed throught init_params attribute.
                                 Default: None (He uniform)
                                
             init_params(dict) - Dictionary of additional parameters for weight initializtion function.
                                 Default: None
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, bias_behaviour='-',
                 dilation=1, rho=0.5, act_fn=None, act_fn_params=None,
                 weight_init=None, init_params=None):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super(ADNNConvTranspose2dCell, self).__init__(in_channels, out_channels, kernel_size,
                                                      stride, padding, dilation, True, output_padding, 
                                                      groups, bias, bias_behaviour, rho, act_fn, act_fn_params,
                                                      weight_init, init_params)


    def _operator(self, x, weight, output_size):
        # type: (Tensor, Optional[List[int]]) -> Tensor
        output_padding = self._output_padding(x, output_size, self.stride,
                                              self.padding, self.kernel_size, self.dilation)
        res = F.conv_transpose2d(x, weight, None, self.stride, self.padding,
                                  output_padding, self.groups, self.dilation)
        return self._extra_padding(res,output_size)



    def _inverse_operator(self, x, weight, output_size):
        return self._inverse_extra_padding(F.conv2d(x, weight, None, self.stride,
                                                    tup_sub_int(self.padding,
                                                                bool(self.extra_padding_flag)),
                                                    self.dilation, self.groups))
    




class ADNNConvTranspose3dCell(_ADNNConvNdCell):
    """
        Applies a 3D transposed convolution(deconvolution) over an input signal composed of several input channels
        Inherits from _ADNNConvNdCell.
        
        Attributes:
             in_channels(int) - Number of channels in the input tensor.
             
             out_channels(int) - Number of channels produced by the convolution.
             
             kernel_size(int) - Size of the convolving kernel.
             
             stride(int or tuple) - Stride of the convolution.
                                    Default: 1
             
             padding (int or tuple) - Zero-padding added to both sides of the input.
                                      Default: 0
             
             output_padding(int or tuple) - Additional size added to one side of the output shape.
                                            Default: 0
             
             dilation (int or tuple) - Spacing between kernel elements.
                                       Default: 1
             
             groups (int) - Number of blocked connections from input channels to output channels.
                            Default: 1
             
             
             padding_mode(str) - Used only for extra output padding in transposed convolution if kernel_size is even.
                                 Can take following values ['constant', 'reflect', 'replicate'].
                                 Default: 'constant'
                                 
            
             bias(bool) - Flag for including bias weight in computation. Note: follows the original article[1] bias is
                          substracted from pre-activation. However, this can be changed via bias_behaviour attribute.
                          Default: True.
                        
             bias_behaviour(str) - Can take following values ['-','+'], substruction or adding correspondingly.
                                   Default: '-' (substruction)
                                  
             rho(float) - Penality factor in augmented lagrangian. See details in [1].
                          Default: 0.5
                         
             act_fn(torch.nn.Module) - Activation function built above PyTorch Module.
                                       Default: None (linear activation)
                                      
             act_fn_param(dict) - Dictionary of parameters for activation function.
                                  Default: None
                                 
             weight_init(func) - Function for weight initialization, takes weight as input and change it inplace.
                                 Additional arguments can be passed throught init_params attribute.
                                 Default: None (He uniform)
                                
             init_params(dict) - Dictionary of additional parameters for weight initializtion function.
                                 Default: None
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, bias_behaviour='-',
                 dilation=1, rho=0.5, act_fn=None, act_fn_params=None,
                 weight_init=None, init_params=None):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        output_padding = _triple(output_padding)
        super(ADNNConvTranspose3dCell, self).__init__(in_channels, out_channels, kernel_size,
                                                      stride, padding, dilation, True, output_padding,
                                                      groups, bias, bias_behaviour, rho, act_fn, act_fn_params,
                                                      weight_init, init_params)


    def _operator(self, x, weight, output_size):
        # type: (Tensor, Optional[List[int]]) -> Tensor
        output_padding = self._output_padding(x, output_size, self.stride,
                                              self.padding, self.kernel_size, self.dilation)
        res = F.conv_transpose3d(x, weight, None, self.stride, self.padding,
                                  output_padding, self.groups, self.dilation)
        return self._extra_padding(res,output_size)



    def _inverse_operator(self, x, weight, output_size):
        return self._inverse_extra_padding(F.conv3d(x, weight, None, self.stride,
                                                    tup_sub_int(self.padding,
                                                                bool(self.extra_padding_flag)),
                                                    self.dilation, self.groups))
