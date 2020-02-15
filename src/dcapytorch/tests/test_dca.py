import pytest
from dcapytorch import *  
import torch
import torch.utils.data
from torch.nn.modules.utils import _single, _pair, _triple
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME
import  numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm

def get_mnist(batch_size):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    return train_loader

def train(model, device, train_loader, loss, optimizer,  epoch, target_name='data', profiler=False):
    model.train()
    with tqdm(train_loader) as pbar:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = loss(model(data),locals()[target_name])
            output.backward()
            optimizer.step()
            if batch_idx % 20:
                pbar.set_postfix(cur_loss='{:.3f}'.format(output.item()))
                if profiler:
                    mem_report()
            pbar.update()



def get_cells_bundles(cell_compatibilities):
    cells_bundles = []
    tmp = []
    prev = 2
    for i, cell in enumerate(cell_compatibilities):
        if cell == 1:
            tmp.append(i)
            prev = cell
        else:
            tmp.append(i)
            if prev != cell:
                prev = cell
            cells_bundles.append(tmp)
            tmp = []
    return cells_bundles, tmp


def test_init_ADNN():
    cell_flatten = Flatten()
    cell_linear = ADNNLinearCell(256, 256,act_fn=torch.nn.ReLU,weight_init=torch.nn.init.eye_)
    cell_conv1d = ADNNConv1dCell(64, 64, kernel_size=9, 
                                 stride=2, padding=0, groups=64,dilation=2)
    cell_reshape = Reshape(16,16)
    unroll=0
    cells = cell_flatten
    adnn = ADNN(cells,unroll)
    print('Init with one cell - OK')
    cells = (cell_linear,cell_linear)
    adnn = ADNN(cells,unroll)
    print('Init with tuple of cells - OK')
    cells =  torch.nn.Sequential(cell_linear,cell_linear)
    adnn = ADNN(cells,unroll)
    print('Init with Sequential - OK')
    cells = (cell_flatten, cell_linear, cell_linear,cell_flatten,cell_linear,cell_reshape)
    comp = [1,2,2,1,2,1]
    bundles = [[0,1],[2],[3,4]]
    ls = [5]
    adnn = ADNN(cells, unroll)
    assert [cell.adnn_compatibility for cell in adnn.cells] == comp, 'ADNN compatibility check fails'
    assert adnn.cells_bundles == bundles and adnn.last_stand == ls, 'Separating on bundles fails'
    cells = (cell_linear, cell_linear,cell_flatten,cell_linear)
    comp = [2,2,1,2]
    bundles = [[0],[1],[2,3]]
    ls = None
    adnn = ADNN(cells, unroll)
    assert [cell.adnn_compatibility for cell in adnn.cells] == comp, 'ADNN compatibility check fails'
    assert adnn.cells_bundles == bundles and adnn.last_stand == ls, 'Separating on bundles fails'
    cells = (cell_flatten,cell_flatten, cell_linear, cell_linear, cell_linear,
             cell_flatten, cell_flatten,cell_linear)
    comp = [1,1,2,2,2,1,1,2]
    bundles = [[0,1,2],[3],[4],[5,6,7]]
    ls = None
    adnn = ADNN(cells, unroll)
    assert [cell.adnn_compatibility for cell in adnn.cells] == comp, 'ADNN compatibility check fails'
    assert adnn.cells_bundles == bundles and adnn.last_stand == ls, 'Separating on bundles fails'
    print('Consistency checking - OK')
    
    

def test_ADNN_run_simple_autoencoder():
    unroll=0
    cells=[Flatten(),
           ADNNLinearCell(784, 100,act_fn=torch.nn.ReLU),
           ADNNLinearCell(100, 100,act_fn=torch.nn.ReLU),
           ADNNLinearCell(100, 784),
           Reshape(1,28,28)
          ]
    adnn = ADNN(cells,unroll)
    device = torch.device("cpu")
    if CUDA_HOME:
        device = torch.device("cuda")
    epochs = 1
    batch_size = 32
    model = adnn.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    train_loader = get_mnist(batch_size)
    loss = torch.nn.MSELoss()
    print('Summary:\n\nNetArchitecture: {net_arch}\n\nOptimizer: {opt}\n\nLoss: {loss}\n\n'.format(net_arch=adnn,opt=optimizer,loss=loss))
    for epoch in tqdm(range(1, epochs + 1),desc='Training without unroll'):
        train(model, device, train_loader, loss, optimizer, epoch)
    print('First subtest passed')
    adnn.reset_parameters()
    unroll = 5
    adnn.unroll = unroll + 1
    for epoch in tqdm(range(1, epochs + 1),desc='Training with unroll={}'.format(adnn.unroll-1)):
        train(model, device, train_loader, loss, optimizer, epoch)
    print('Second subtest passed') 
    

def test_ADNN_run_convolutional_autoencoder():
    unroll=0    
    cells=[ADNNConv2dCell(1, 16, kernel_size=3, stride=2,padding=1,act_fn=torch.nn.ReLU),
           ADNNConv2dCell(16, 8, kernel_size=3, stride=2,padding=1,act_fn=torch.nn.ReLU),
           ADNNConv2dCell(8, 8, kernel_size=3, stride=2,act_fn=torch.nn.ReLU),
           Flatten(),
           ADNNLinearCell(72,72,act_fn=torch.nn.ELU),
           Reshape(8,3,3),
           ADNNConvTranspose2dCell(8,8,kernel_size=3, stride=2,act_fn=torch.nn.ReLU),
           ADNNConvTranspose2dCell(8,16,kernel_size=3, stride=2,padding=1,act_fn=torch.nn.ReLU),
           ADNNConvTranspose2dCell(16,1,kernel_size=3, stride=2,output_padding=1)
          ]
    adnn = ADNN(cells,unroll)
    device = torch.device("cpu")
    if CUDA_HOME:
        device = torch.device("cuda")
    epochs = 1
    batch_size = 32
    model = adnn.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    train_loader = get_mnist(batch_size)
    loss = torch.nn.MSELoss()
    print('Summary:\n\nNetArchitecture: {net_arch}\n\nOptimizer: {opt}\n\nLoss: {loss}\n\n'.format(net_arch=adnn,opt=optimizer,loss=loss))
    for epoch in tqdm(range(1, epochs + 1),desc='Training without unroll'):
        train(model, device, train_loader, loss, optimizer, epoch)
    print('First subtest passed')
    adnn.reset_parameters()
    unroll = 20
    adnn.unroll = unroll + 1
    for epoch in tqdm(range(1, epochs + 1),desc='Training with unroll={}'.format(adnn.unroll-1)):
        train(model, device, train_loader, loss, optimizer, epoch)
    print('Second subtest passed')    


def test_ADNN_run_incorporated():
    adnn_decoder = ADNN(cells=[Reshape(8,3,3),
                       ADNNConvTranspose2dCell(8,8,kernel_size=3, stride=2,act_fn=torch.nn.ReLU),
                       ADNNConvTranspose2dCell(8,16,kernel_size=3, stride=2,padding=1,act_fn=torch.nn.ReLU),
                       ADNNConvTranspose2dCell(16,1,kernel_size=3, stride=2,output_padding=1)
                       ],unroll=0)
    net = torch.nn.Sequential(torch.nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=2),
                              torch.nn.ReLU(),
                              torch.nn.Conv2d(16, 8, kernel_size=3, padding=1, stride=2),
                              torch.nn.ReLU(),
                              torch.nn.Conv2d(8, 8, kernel_size=3, stride=2),
                              torch.nn.ReLU(),
                              Flatten(),
                              torch.nn.Linear(72, 72),
                              torch.nn.ELU(),
                              adnn_decoder
                             )
   
    device = torch.device("cpu")
    if CUDA_HOME:
        device = torch.device("cuda")
    epochs = 1
    batch_size = 32
    model = net.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    train_loader = get_mnist(batch_size)
    loss = torch.nn.MSELoss()
    print('Summary:\n\nNetArchitecture: {net_arch}\n\nOptimizer: {opt}\n\nLoss: {loss}\n\n'.format(net_arch=net,opt=optimizer,loss=loss))
    for epoch in tqdm(range(1, epochs + 1),desc='Training without unroll'):
        train(model, device, train_loader, loss, optimizer, epoch)
    print('First subtest passed')
    for cell in model:
        if hasattr(cell,'reset_parameters'):
            cell.reset_parameters()
    unroll = 20
    net[-1].unroll = unroll + 1
    for epoch in tqdm(range(1, epochs + 1),desc='Training with unroll={}'.format(net[-1].unroll-1)):
        train(model, device, train_loader, loss, optimizer, epoch)
    print('Second subtest passed')    
