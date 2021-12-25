
# Implementation of 3D-patch-based MESC-SD in pytorch

import torch
import torch.nn as nn


class Linear_2D(nn.Module):
    def __init__(self, in_features_2, in_features_1, out_features_2, out_features_1, bias_1=True, bias_2=True):
        super(Linear_2D, self).__init__()
        
        self.in_features_1 = in_features_1
        self.in_features_2 = in_features_2
        self.out_features_1 = out_features_1
        self.out_features_2 = out_features_2
        self.bias_1 = bias_1
        self.bias_2 = bias_2
        self.linear_1 = nn.Linear(in_features=self.in_features_1, out_features=self.out_features_1, bias=self.bias_1)
        self.linear_2 = nn.Linear(in_features=self.in_features_2, out_features=self.out_features_2, bias=self.bias_2)

    def forward(self,x):
        y = x.permute(0,2,1)
        y = self.linear_1(y)
        y = y.permute(0,2,1)
        y = self.linear_2(y)
        return y

class SD_LSTM(nn.Module):
    def __init__(self, in_features_2, in_features_1=27, hid_features_2=300, hid_features_1=300, relu_th = 0.01):
        super(SD_LSTM, self).__init__()
        
        self.in_features_1 = in_features_1
        self.in_features_2 = in_features_2
        self.hid_features_1 = hid_features_1
        self.hid_features_2 = hid_features_2
        self.relu_th = relu_th
        self.linear_2d_cx = Linear_2D(in_features_2 = self.hid_features_2, in_features_1 = self.hid_features_1, out_features_2= self.hid_features_2, out_features_1 = self.hid_features_1)
        self.linear_2d_cy = Linear_2D(in_features_2 = self.in_features_2, in_features_1 = self.in_features_1, out_features_2= self.hid_features_2, out_features_1 = self.hid_features_1)
        self.linear_2d_fx = Linear_2D(in_features_2 = self.hid_features_2, in_features_1 = self.hid_features_1, out_features_2= self.hid_features_2, out_features_1 = self.hid_features_1)
        self.linear_2d_fy = Linear_2D(in_features_2 = self.in_features_2, in_features_1 = self.in_features_1, out_features_2= self.hid_features_2, out_features_1 = self.hid_features_1)
        self.linear_2d_gx = Linear_2D(in_features_2 = self.hid_features_2, in_features_1 = self.hid_features_1, out_features_2= self.hid_features_2, out_features_1 = self.hid_features_1)
        self.linear_2d_gy = Linear_2D(in_features_2 = self.in_features_2, in_features_1 = self.in_features_1, out_features_2= self.hid_features_2, out_features_1 = self.hid_features_1)
        self.sigmoid_f = nn.Sigmoid()
        self.sigmoid_g = nn.Sigmoid()
        self.relu_thresholded = nn.Threshold(threshold=self.relu_th, value=0)

    def forward(self,y,x,c):
        cy = self.linear_2d_cy(y)
        cx = self.linear_2d_cx(x)
        c_gate = cy - cx + x
        fy = self.linear_2d_fy(y)
        fx = self.linear_2d_fx(x)
        f_gate = self.sigmoid_f(fy+fx)
        gy = self.linear_2d_gy(y)
        gx = self.linear_2d_gx(x)
        g_gate = self.sigmoid_g(gy+gx)
        c_out = f_gate * c + g_gate * c_gate
        x_out = self.relu_thresholded(c_out)
        return x_out, c_out




class MESC_SD(nn.Module):
    def __init__(self, n_channels, n_voxels=27, n_out = 11, n_hidden_1 = 300, n_hidden_2 = 75):

        '''
        n_channels ---> number of input DWIs
        n_voxels   ---> number of voxels in the 3D input patch; default 3x3x3=27
        n_out      ---> number of output diffusion parameters
        n_hidden_1 ---> number of spatial and angular features in the sparse representation
        n_hidden_2 ---> number of spatial and angular features in the modified FC layers

        '''
        
        super(MESC_SD, self).__init__()
        
        self.n_channels = n_channels
        self.n_voxels = n_voxels
        self.n_out = n_out
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.sd_lstm = SD_LSTM(in_features_2=self.n_channels, in_features_1=self.n_voxels, hid_features_2=self.n_hidden_1, hid_features_1=self.n_hidden_1)
        
        self.fc_1 = Linear_2D(in_features_2 = self.n_hidden_1, in_features_1 = self.n_hidden_1, out_features_2= self.n_hidden_2, out_features_1 = self.n_hidden_2)
        self.relu_1 = nn.ReLU()
        self.fc_2 = Linear_2D(in_features_2 = self.n_hidden_2, in_features_1 = self.n_hidden_2, out_features_2= self.n_hidden_2, out_features_1 = self.n_hidden_2)
        self.relu_2 = nn.ReLU()
        self.fc_3 = Linear_2D(in_features_2 = self.n_hidden_2, in_features_1 = self.n_hidden_2, out_features_2= self.n_out, out_features_1 = 1)
        self.relu_3 = nn.ReLU()
        
        
    def forward(self, y):
        x, c = self.init_hidden(y)
        x, c = self.sd_lstm(y,x,c)
        x, c = self.sd_lstm(y,x,c)
        x, c = self.sd_lstm(y,x,c)
        x, c = self.sd_lstm(y,x,c)
        x, c = self.sd_lstm(y,x,c)
        x, c = self.sd_lstm(y,x,c)
        x, c = self.sd_lstm(y,x,c)
        x, c = self.sd_lstm(y,x,c)
        out = self.fc_1(x)
        out = self.relu_1(out)
        out = self.fc_2(out)
        out = self.relu_2(out)
        out = self.fc_3(out)
        # This Relu imposes positivity constraint on the output paremeters
        # Sometimes it makes the training unstable, which can be removed in those cases
        # without affecting the results, significantly
        out = self.relu_3(out)
        return out

    def init_hidden(self,y):
        x_0 = torch.zeros(y.size(0), self.n_hidden_1, self.n_hidden_1, device=y.device)
        c_0 = torch.zeros(y.size(0), self.n_hidden_1, self.n_hidden_1, device=y.device)
        return x_0, c_0
    
    
def main():

    from model_3d import MESC_SD
    
    num_channels = 30
    num_voxels = 27
    batch_size = 16
    MESC_SD_model = MESC_SD(n_channels=num_channels, n_voxels = num_voxels)

    dummy_input = torch.Tensor(torch.rand(batch_size ,num_voxels, num_channels))
    dummy_output = MESC_SD_model(dummy_input)
    print(f'For MESC-SD with input of size {dummy_input.shape}:')
    print(f'The output size is {dummy_output.shape}')

if __name__ == "__main__":
    main()

    
