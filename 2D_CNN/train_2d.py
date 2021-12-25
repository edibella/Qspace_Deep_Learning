
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

# importing 2D dataset and CNN model
sys.path.append("./..")
from Utils.dataset_2d import Dataset_2D
from model_2d import CNN_2D


# Some useful functions
def get_num_param(mdl):
    return sum(p.numel() for p in mdl.parameters() if p.requires_grad)

def RMSE(a1,a2,mask):
    return np.sqrt(np.sum(((a1-a2)*mask)**2))
    

def nRMSE(a1,a_ref,a_mask):
    n_batch = a_ref.shape[0]
    n_ch = a_ref.shape[1]
    b1 = np.reshape(a1,(n_batch, n_ch, -1))
    b1 = b1.transpose(0,2,1).reshape(-1,n_ch)
    b_ref = np.reshape(a_ref,(n_batch,n_ch,-1))
    b_ref = b_ref.transpose(0,2,1).reshape(-1,n_ch)
    b_mask = np.reshape(a_mask,(n_batch,1,-1))
    b_mask = b_mask.transpose(0,2,1).reshape(-1,1)
    nrmse = np.zeros([1,n_ch])

    for n_c in range(n_ch):
        nrmse[0,n_c] =2*(RMSE(b_ref[:,n_c],b1[:,n_c],b_mask[:,0])
                    /RMSE(b_ref[:,n_c],-b_ref[:,n_c],b_mask[:,0]))
            
    return nrmse

def get_score_model(model, data_loader):
    nrmse_list = [] 
    #toggle model to eval mode
    model.eval()
    #turn off gradients since they will not be used here
    with torch.no_grad():             
        for x, y_tg, x_mask  in data_loader:
            x = x.cuda()
            x_mask = x_mask.cuda()
            y = model(x)
            y = y * x_mask
            this_nrmse = nRMSE(y.cpu().detach().numpy(),y_tg.cpu().detach().numpy(), x_mask.cpu().detach().numpy())
            nrmse_list.append(this_nrmse)
             
    nrmse = np.concatenate(nrmse_list, axis = 0)
    return np.mean(nrmse, axis=0)





os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

parser = argparse.ArgumentParser(description="Training 2D-CNN")
parser.add_argument("--sampling", type=str, default='30_DWI', help='Q-space undersampling pattern name')
parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
parser.add_argument("--num_of_channels", type=int, default=30, help="Number of CNN input channels")
parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--data_path", type=str, default='./../Data/', help='Path to the data')
opt = parser.parse_args()

def main():
    
    
    # Load dataset
    print('Loading training dataset ...\n')
    data_path = opt.data_path + opt.sampling +  '/train_2d/'
    dataset_train = Dataset_2D(data_path, data_type = 'train', split = 0.80)
    dataset_val = Dataset_2D(data_path, data_type = 'val', split = 0.80)
    
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
    net = CNN_2D(n_channels = opt.num_of_channels, n_out = 11)
    criterion = nn.L1Loss(reduction='mean')
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
        for x, y_tg, x_mask in train_loader:
            # training step
            
            model.zero_grad()
            optimizer.zero_grad()
            
            x = x.cuda()
            y_tg = y_tg.cuda()
            x_mask = x_mask.cuda()
            y = model(x)
            y = y * x_mask
            
            loss = criterion(y,y_tg)
            loss.backward()
            optimizer.step()
            
            loss_list.append(loss.item())
            
            
            
        
        
        #at the end of each epoch
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
    torch.save(model.state_dict(), model_path + '2D_CNN.pth')
        
        
if __name__ == "__main__":
    main()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
