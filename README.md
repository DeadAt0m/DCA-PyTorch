## Deep Component Analysis for PyTorch

Implementation of Alternating Direction Neural Network(ADNN) from 'Deep Component Analysis via Alternating Direction Neural Networks' article(https://arxiv.org/abs/1803.06407)

The ADNN extends usuall forward pass of networks with additional(recurrent) iterations inspired by Alternating Direction
Method of Multipliers. This makes activation functions work like proximal operators(incorporate any geometric priors into network subspace). Number of additional iterations can be viewed as unroll of recurent network(on scheme), so it linearly increases the training time. However, unlike classic RNN unroll it required calculation of inverse operations(backward projections).

![alt text](https://github.com/DeadAt0m/DCA-PyTorch/raw/master/adnn_scheme.png "ADNN Scheme")

ADNN cell class implements all logic for handling forward pass on a par with inverse operations. 
Next cells are already available: Linear, Conv1D, Conv2D, Conv3D(and transposed versions), however new ones may implemented using provided _ADNNBaseCell interface.

ADNN class does all unroll operations under ADNN cells:

    1. It can be incorporated in any part of your network
    2. Required memory and perfomance are linear depend of unroll value. So, add aditional iterations wisely.
    3. It will work as common forward network if unroll set to zero.



For more details read the original article.

## Requirements

  - PyTorch >= 1.3.0


## Installation

Just run: python setup.py install

## Using
   
   *import dcapytorch* - import all necessary modules: ADNN, ADNN\*Cell
   I recommend the read the classes descriptions in adnn.py and adnn_cell.py. It's well documented.

## Examples
   *./tests/test_dca.py* - contains examples of using ADNN in the wild.

## Testing

Some unit tests provided by next command: python setup.py tests.

