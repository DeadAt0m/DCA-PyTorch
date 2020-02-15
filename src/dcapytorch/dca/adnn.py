import torch

class ADNN(torch.nn.Module):
    '''
        Alternating Direction Neural Networks(ADNN) class does forward operations under ADNN cell or cells,
        and handles activations' storage for them.
        
        Only works with submodules containing adnn_compatibility parameter. For more information see dca/adnn_cell.py
        
        By default during inference the adnn cell works as Sequential module, however this can be controlled
        via force_unrolled_inference parameter
    '''
    
    
    __constants__ = ['unroll']
    
    def __init__(self,cells,unroll=0):
        '''
            Arguments:
                cells - list, tuple or torch.nn.Sequential instance containing modules with adnn_compatibility parameter.
                unroll - integer, number of additional recurrent forward passes. Zero unroll is equals to common Sequential module,
                         however it keeps all layers activations and pre-activation(before activation function layer).Default: 0
            
        '''
        super(ADNN, self).__init__()
        self.unroll = unroll + 1
        self.cells = cells
        self.force_unrolled_inference = False
        self._check_cells_consistency()
        

    def _check_cells_consistency(self):
        self.cells_bundles = []
        #check if cells is just cell and and wrap it in Sequential
        if hasattr(self.cells, 'adnn_compatibility'):
            try:
                self.cells = torch.nn.Sequential(self.cells)
            except:
                raise ValueError('Cell cannot be wraped into Sequential container automatically') 
        if type(self.cells) == list or type(self.cells) == tuple:
            try:
                self.cells = torch.nn.Sequential(*self.cells)
            except:
                raise ValueError('Cell cannot be wraped into Sequential container automatically',
                                 'because it required additional params for init') 
        #check if it Sequential
        if type(self.cells) != torch.nn.modules.container.Sequential:
            raise ValueError('Cells not in Sequential container')
        #continue checking and forming bundles
        tmp_list = []
        tmp_prev = 2    
        for i, cell in enumerate(self.cells):
            if not hasattr(cell, 'adnn_compatibility') or  not( cell.adnn_compatibility in [1,2]):
                raise ValueError('{} is not allowed to use in ADNN. Due to missing compatibility parameter')
            if cell.adnn_compatibility == 1:
                tmp_list.append(i)
                tmp_prev = cell.adnn_compatibility
            else:
                tmp_list.append(i)
                if tmp_prev != cell.adnn_compatibility:
                    tmp_prev = cell.adnn_compatibility
                self.cells_bundles.append(tmp_list)
                tmp_list = []
        self.last_stand = tmp_list if tmp_list else None
            
    def reset_parameters(self):
        for cell in self.cells:
            cell.reset_parameters()
  
    def _get_cell(self,i):
        return self.cells._modules[str(i)]       
    
    def _reverse_pre_activation(self, input, cells):
        target_cell = self._get_cell(cells[-1])
        out = target_cell._inverse_operator(input,
                                            target_cell.weight,
                                            target_cell.input_shape)
        #reverse weak cells transformations
        if len(cells) > 1:
            for i in reversed(cells[:-1]):
                out = self._get_cell(i).reverse(out)
        return out
        
            
    def forward(self,input,DEBUG=False):
        act_storage = [{} for elem in self.cells_bundles]
        
        for step in range(self.unroll if self.training or self.force_unrolled_inference else 1):
            for i,bundle in enumerate(self.cells_bundles):
                args_dict = {}
                args_dict['unroll_step'] = step
                # when step is zero prev_activation and activation don't count
                
                args_dict['input'] = input if (i==0) and (step==0) else act_storage[i-(step==0)]['activation']
                args_dict['pre_activation'] = input if (i==0) and (step==0) else act_storage[i-(step==0)]['pre_activation']
                args_dict['prev_activation'] = input if (i == 0) or (step==0) else act_storage[i-1]['activation']
                args_dict['old_activation'] = None
                if ((step!=0) and (i < len(self.cells_bundles) - 1)):
                    args_dict['old_activation'] = self._reverse_pre_activation(act_storage[i+1]['pre_activation'], cells=self.cells_bundles[i+1])
                
                #debug flags
                prev_act_tf_flag = False
                pre_act_tf_flag = False
                input_act_tf_flag = False
                
                
                if len(bundle) > 1  and (step==0):
                    if bundle[0] == 0:
                        args_dict['pre_activation'] = None
                    for elem in bundle[:-1]:
                        res = self._get_cell(elem)(**args_dict)
                        if type(res) == tuple:
                            args_dict['pre_activation'], args_dict['input']  = res
                            pre_act_tf_flag = True
                            input_act_tf_flag = True
                        else:
                            args_dict['input'] = res
                            input_act_tf_flag = True
                elif  len(bundle) > 1:
                    for elem in bundle[:-1]:
                        args_dict['prev_activation'] = self._get_cell(elem)(args_dict['prev_activation']) 
                    prev_act_tf_flag = True   
                
                args_dict['activation'] = args_dict['input']
                if args_dict['pre_activation'] is None:
                    args_dict['pre_activation'] = args_dict['input']

                
                if DEBUG:
                    frmt_str = 'Step:{unroll_step}, Bundle №:{bundle_num}, Bundle content: {bundle_content},\n' +\
                               'args_dict["input"]={input},\n' +\
                               'args_dict["pre_activation"]={pre_activation},\n' +\
                               'args_dict["prev_activation"]={prev_activation},\n' +\
                               'args_dict["activation"]={activation},\n' +\
                               'args_dict["unroll_step"]={unroll_step},\n' +\
                               'Is prev_activation was transformed? {prev_act_tf_flag},\n' +\
                               'Is pre_activation was transformed? {pre_act_tf_flag},\n' +\
                               'Is input was transformed? {input_act_tf_flag},\n'
                    print(frmt_str.format(unroll_step=step,bundle_num=i,bundle_content=bundle,
                                          input='input' if (i==0) and (step==0) else 'act_storage[{}]["activation"]'.format(i-(step==0)),
                                          pre_activation='input' if (i==0) and (step==0) else 'act_storage[{}]["pre_activation"]'.format(i-(step==0)),
                                          prev_activation='input' if (i==0) or (step==0) else 'act_storage[{}]["activation"]'.format(i-1),
                                          activation='input' if (i==0) and (step==0) else 'act_storage[{}]["activation"]'.format(i-(step==0)),
                                          prev_act_tf_flag = prev_act_tf_flag,
                                          pre_act_tf_flag = pre_act_tf_flag,
                                          input_act_tf_flag = input_act_tf_flag
                                         ))
                    
                act_storage[i]['pre_activation'], act_storage[i]['activation'] = self._get_cell(bundle[-1])(**args_dict)
        if self.last_stand:
            for elem in self.last_stand:
                act_storage[-1]['activation'] = self._get_cell(elem)(act_storage[-1]['activation']) 
        out =  act_storage[-1]['activation'].clone() 
        del act_storage
        return  out    
                
                
                
            
        
        
        
        
      