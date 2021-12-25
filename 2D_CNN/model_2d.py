
# implementation of 2D-CNN model in pytorch

import torch
import torch.nn as nn


class Res_Block(nn.Module):
    def __init__(self, C_in, C_out):
        super(Res_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = C_in, out_channels = C_out, kernel_size = 1, padding = 0)
        
        self.relu1= nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels = C_out, out_channels = C_out, kernel_size = 3, padding = 1)
        
        self.relu2= nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels = C_out, out_channels = C_out, kernel_size = 3, padding = 1)
        
        self.relu3= nn.ReLU()
        
    def forward(self, x):
        y1 = self.conv1(x)
        y1 = self.relu1(y1)
        y2 = self.conv2(y1)
        y2 = self.relu2(y2)
        y2 = self.conv3(y2)
        y2 = self.relu3(y2)
        y = torch.add(y2,y1)
        return y

class Out_Block(nn.Module):
    def __init__(self, C_in, C_out):
        super(Out_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels = C_in, out_channels = C_out, kernel_size = 1, padding = 0)
        
    def forward(self, x):
        y = self.conv(x)
        return y


class CNN_2D(nn.Module):
    def __init__(self, n_channels, n_out = 11, res_hiddens = [128, 256]):

        '''
        n_channels  ---> number of input DWIs
        n_out       ---> number of output diffusion parameters
        res_hiddens ---> number of hidden features in res blocks

        '''
        super(CNN_2D, self).__init__()
        layers = []
        layers.append(Res_Block(n_channels,res_hiddens[0]))
        for ii in range(1,len(res_hiddens)):
            layers.append(Res_Block(res_hiddens[ii-1],res_hiddens[ii]))
              
        layers.append(Out_Block(res_hiddens[ii],n_out))       
                      
        self.simple_cnn = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.simple_cnn(x)
        return out
    
    
def main():
    
    from model_2d import CNN_2D
    
    num_channels = 30
    batch_size = 16
    CNN_model = CNN_2D(num_channels)
    dummy_input = torch.Tensor(torch.rand(batch_size , num_channels, 140, 140))
    dummy_output = CNN_model(dummy_input)
    print(f'For 2D-CNN with input of size {dummy_input.shape}:')
    print(f'The output size is {dummy_output.shape}')
    


if __name__ == "__main__":
    main()

    
