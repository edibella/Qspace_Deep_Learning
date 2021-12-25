
# Implementation of 1D-qDL training
# For the github version the trainig data are not available

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# importing 1D dataset and qDL model
sys.path.append("./..")
from Utils.dataset_1d import Dataset_1D
from model_1d import qDL


# some useful functions
def get_num_param(mdl):
    return sum(p.numel() for p in mdl.parameters() if p.requires_grad)

def RMSE(a1,a2):
    return np.sqrt(np.sum((a1-a2)**2))

def nRMSE(a1,a_ref):
    n_ch = a_ref.shape[1]
    nrmse = np.zeros([1,n_ch])
    
    for n_c in range(n_ch):
        nrmse[0,n_c] = 2*RMSE(a_ref[:,n_c],a1[:,n_c])/RMSE(a_ref[:,n_c],-a_ref[:,n_c])
    return nrmse

def get_score_model(model, data_loader):
    nrmse_list = [] 
    #toggle model to eval mode
    model.eval()
    #turn off gradients since they will not be used here
    with torch.no_grad():             
        for x, y_tg  in data_loader:
            x = x.cuda()
            y = model(x)
            this_nrmse = nRMSE(y.cpu().detach().numpy(),y_tg.cpu().detach().numpy())
            nrmse_list.append(this_nrmse)
             
    nrmse = np.concatenate(nrmse_list, axis = 0)
    return np.mean(nrmse, axis=0)





os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

parser = argparse.ArgumentParser(description="Training 1D-qDL")
parser.add_argument("--sampling", type=str, default='30_DWI', help='Q-space undersampling pattern name')
parser.add_argument("--batch_size", type=int, default=10000, help="Training batch size")
parser.add_argument("--num_of_channels", type=int, default=30, help="Number of qDL input channels")
parser.add_argument("--num_of_layers", type=int, default=3, help="Number of qDL layers")
parser.add_argument("--num_of_hidden", type=int, default=150, help="Number of hidden nodes in qDL")
parser.add_argument("--drop_out", type=float, default=0.1, help="Drop out probability in qDL")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--data_path", type=str, default='./../Data/', help='Path to the data')
opt = parser.parse_args()

def main():
    
    
    # Load dataset
    print('Loading training dataset ...\n')
    data_path = opt.data_path + opt.sampling +  '/train_1d/'
    dataset_train = Dataset_1D(data_path, data_type = 'train', split = 0.80)
    dataset_val = Dataset_1D(data_path, data_type = 'val', split = 0.80)
    
    # training data loader
    device_ids = [0, 1, 2, 3]
    b_size = len(device_ids) * opt.batch_size
    train_loader = DataLoader(dataset=dataset_train, num_workers=32, batch_size=b_size, shuffle=True)
    print(f'# of training samples: {len(dataset_train)}')
    print(f'# of training batches: {len(train_loader)}')
    
    # validation data loader
    val_loader = DataLoader(dataset=dataset_val, num_workers=32, batch_size=b_size, shuffle=False)
    print(f'# of validation samples: {len(dataset_val)}')
    print(f'# of validation batches: {len(val_loader)}')
    
    
    # Build model
    net = qDL(n_channels = opt.num_of_channels, n_out = 11, n_layers=opt.num_of_layers, n_hidden = opt.num_of_hidden, drop_out = opt.drop_out)
    criterion = nn.MSELoss(reduction='mean')
    print(f'# of paremeters: {get_num_param(net)}')
    
    # Move to GPU
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()
    
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    
    # training
    running_loss = 0
    running_nrmse = np.zeros(11)
    for epoch in tqdm(range(opt.epochs)):
        
        model.train()
        loss_list = []
        for x, y_tg in train_loader:
            # training step
            
            model.zero_grad()
            optimizer.zero_grad()
            
            x = x.cuda()
            y_tg = y_tg.cuda()
            y = model(x)
            
            loss = criterion(y,y_tg)
            loss.backward()
            optimizer.step()
            
            loss_list.append(loss.item())
            
            
            
        
        
        # at the end of each epoch
        print('Epoch: ' + str(epoch+1) + ' out of ' + str(opt.epochs)) 
        print('Loss: ' + str(np.mean(loss_list)))
        running_loss += np.mean(loss_list)
            
            
        # validation accuracy
        nrmse_val = get_score_model(model, val_loader)
        running_nrmse += nrmse_val
        print('Validation Accuracy in nRMSE: ' + str(nrmse_val))   
        val_metric = np.sum(nrmse_val)
        scheduler.step(val_metric)
        print()
        
        
    # save model
    model_path = opt.data_path + opt.sampling +  '/Models/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(model.state_dict(), model_path + '1D_qDL.pth')
        
        
if __name__ == "__main__":
    main()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
