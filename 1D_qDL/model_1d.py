
# implementation of 1D-qDL model in pytorch


import torch
import torch.nn as nn

class qDL(nn.Module):
    def __init__(self, n_channels, n_out = 11, n_layers=3, n_hidden = 150, drop_out = 0.1):
        
        '''
        n_channels ---> number of input DWIs
        n_out      ---> number of output diffusion parameters
        n_layers   ---> number of hidden layers
        n_hidden   ---> number of hidden nodes
        drop_out   ---> dorpout probability

        '''

        super(qDL, self).__init__()
        
        layers = []
        layers.append(nn.Linear(in_features=n_channels, out_features=n_hidden, bias=True))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=drop_out))
        
        for _ in range(n_layers-1):
            layers.append(nn.Linear(in_features=n_hidden, out_features=n_hidden, bias=True))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=drop_out))
            
            
        layers.append(nn.Linear(in_features=n_hidden, out_features=n_out, bias=True))
        layers.append(nn.ReLU())
        self.q_DL = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.q_DL(x)
        return out
    
    
def main():
    from model_1d import qDL
    
    num_channels = 30
    batch_size = 16
    qDL_model = qDL(num_channels)
    dummy_input = torch.Tensor(torch.rand(batch_size , num_channels))
    dummy_output = qDL_model(dummy_input)
    print(f'For 1D-qDL with input of size {dummy_input.shape}:')
    print(f'The output size is {dummy_output.shape}')
    

if __name__ == "__main__":
    main()

    
