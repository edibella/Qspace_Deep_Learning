

# Implementation of 2D-CNN testing
# For the github version just one test dataset is provided

import json
import os
import sys
import h5py
import argparse

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

# importing HCPA data class and CNN model
sys.path.append("./..")
from Utils.HCPA_data_class import HCPA_Data
from model_2d import CNN_2D

# some useful functions
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




os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

parser = argparse.ArgumentParser(description="2D-CNN testing")
parser.add_argument("--sampling", type=str, default='30_DWI', help='Q-space undersampling pattern name')
parser.add_argument("--batch_size", type=int, default=32, help="Testing batch size")
parser.add_argument("--num_of_channels", type=int, default=30, help="Number of CNN input channels")
parser.add_argument("--data_path", type=str, default='./../Data/', help='Path to the data')
parser.add_argument("--test_cases", type=list, default=['M_56_Strk'], help='List of test subjects ids')
opt = parser.parse_args()



def main():
    diff_methods = ['fa','md','rd','ak','mk','rk','odi','fiso','ficvf','v_int','lambda']
    diff_maps = ['dti_FA.nii.gz','dti_MD.nii.gz','dti_RD.nii.gz',
                         'dki_AK.nii.gz','dki_MK.nii.gz','dki_RK.nii.gz',
                         'noddi_ODI.nii.gz','noddi_FISO.nii.gz','noddi_FICVF.nii.gz',
                         'smt_v_int.nii.gz','smt_lambda.nii.gz']
    
    for subject_id in opt.test_cases:
        # Load the subject data
        
        # Loading mean and max values from training data
        train_path = opt.data_path + opt.sampling +  '/train_2d/'
        x_maxs = np.array(h5py.File(train_path+'x_train_maxs.h5','r')['x_maxs'])
        x_means = np.array(h5py.File(train_path+'x_train_means.h5','r')['x_means'])
        y_means = np.array(h5py.File(train_path+'y_train_means.h5','r')['y_means'])


        print(f'Loading {subject_id} data with {opt.sampling} sampling ...\n')
        hcpa_data = HCPA_Data(opt.data_path, sampling=opt.sampling, subject_id=subject_id)
        data_dwi = hcpa_data.get_dwi()
        data_fa = hcpa_data.get_fa()[:,:,:,None]
        data_md = hcpa_data.get_md()[:,:,:,None]
        data_rd = hcpa_data.get_rd()[:,:,:,None]
        data_ak = hcpa_data.get_ak()[:,:,:,None]
        data_mk = hcpa_data.get_mk()[:,:,:,None]
        data_rk = hcpa_data.get_rk()[:,:,:,None]
        data_odi = hcpa_data.get_odi()[:,:,:,None]
        data_fiso = hcpa_data.get_fiso()[:,:,:,None]
        data_ficvf = hcpa_data.get_ficvf()[:,:,:,None]
        data_v_int = hcpa_data.get_v_int()[:,:,:,None]
        data_lambda = hcpa_data.get_lambda()[:,:,:,None]
        
        data_mask = hcpa_data.get_mask()[:,:,:,None]
        
        dim0, dim1, n_slices, n_channels = data_dwi.shape
    
    
        y_fa = data_fa.transpose(2,3,0,1)
        y_md = data_md.transpose(2,3,0,1)
        y_rd = data_rd.transpose(2,3,0,1)
        y_ak = data_ak.transpose(2,3,0,1)
        y_mk = data_mk.transpose(2,3,0,1)
        y_rk = data_rk.transpose(2,3,0,1)
        y_odi = data_odi.transpose(2,3,0,1)
        y_fiso = data_fiso.transpose(2,3,0,1)
        y_ficvf = data_ficvf.transpose(2,3,0,1)
        y_v_int = data_v_int.transpose(2,3,0,1)
        y_lambda = data_lambda.transpose(2,3,0,1)
        
        x_mask = data_mask.transpose(2,3,0,1)
        x_dwi = data_dwi.transpose(2,0,1,3)
        x_dwi -= x_means
        x_dwi /= x_maxs
        x_dwi = x_dwi.transpose(0,3,1,2)
    
     
        y_tg = np.concatenate((y_fa, y_md, y_rd, y_ak, y_mk, y_rk, y_odi, y_fiso, y_ficvf, y_v_int, y_lambda), axis=1)
        y_tg = y_tg.transpose(0,2,3,1)
        y_tg /= np.exp(np.floor(np.log(y_means))) 
        y_tg = y_tg.transpose(0,3,1,2)


    
        # Loading the model
        print('Loading model ...\n')
        model_path = opt.data_path + opt.sampling +  '/Models/'
        net = CNN_2D(n_channels = opt.num_of_channels, n_out = 11)
        device_ids = [0,1,2,3]
        b_size = len(device_ids) * opt.batch_size
        
        model = nn.DataParallel(net, device_ids=device_ids).cuda()
        model.load_state_dict(torch.load(model_path + '2D_CNN.pth'))
        model.eval()
    
    
   
        # Processing the test data
        print('Processing the test data ...\n')
    
        iterations = x_dwi.shape[0] // b_size
        last_batch_size = x_dwi.shape[0] % b_size
    
        nrmse_list = []
        y_est = np.zeros(y_tg.shape)
        
    
        with torch.no_grad():

            for ii in range(iterations):
                x = x_dwi[ii*b_size:(ii+1)*b_size,:,:,:]
                x = torch.FloatTensor(x)
                y_target = y_tg[ii*b_size:(ii+1)*b_size,:,:,:]
                y_target = torch.FloatTensor(y_target)
                x_msk = x_mask[ii*b_size:(ii+1)*b_size,:,:,:]
                x_msk = torch.FloatTensor(x_msk)
            
            
                x = x.cuda()
                y_target = y_target.cuda()
                x_msk = x_msk.cuda()
                y = model(x)
                y = y * x_msk
        
                y = y.cpu().detach().numpy()
                y_target = y_target.cpu().detach().numpy()
                x_msk = x_msk.cpu().detach().numpy()
        
                nrmse_list.append(nRMSE(y,y_target,x_msk))
                y_est[ii*b_size:(ii+1)*b_size,:,:,:] = y
                
    
  
            x = x_dwi[iterations*b_size:iterations*b_size+last_batch_size,:,:,:]
            x = torch.FloatTensor(x)
            y_target = y_tg[iterations*b_size:iterations*b_size+last_batch_size,:,:,:]
            y_target = torch.FloatTensor(y_target)
            x_msk = x_mask[iterations*b_size:iterations*b_size+last_batch_size,:,:,:]
            x_msk = torch.FloatTensor(x_msk)
        
            x = x.cuda()
            y_target = y_target.cuda()
            x_msk = x_msk.cuda()
            y = model(x)
            y = y * x_msk
        
            y = y.cpu().detach().numpy()
            y_target = y_target.cpu().detach().numpy()
            x_msk = x_msk.cpu().detach().numpy()
        
            nrmse_list.append(nRMSE(y,y_target,x_msk))
            y_est[iterations*b_size:iterations*b_size+last_batch_size,:,:,:] = y
            
    
            nrmse_test = np.mean(np.concatenate(nrmse_list, axis = 0),axis=0)
            print('Testing Accuracy in nRMSE: ' + str(nrmse_test))
            print()
        
            # Save testing results
            y_est = y_est.transpose(0,2,3,1)
            y_est *= np.exp(np.floor(np.log(y_means))) 
            y_est = y_est.transpose(1,2,0,3)
            
        
        
            out_path = opt.data_path + opt.sampling +  '/test_2d/' + subject_id + '/'
            if not os.path.exists(out_path):
                os.makedirs(out_path)
        
            for jj in range(len(diff_maps)):
                 get_method = getattr(hcpa_data, 'get_'+ diff_methods[jj])
                 data_ = get_method(return_nifti=True)
                 data_ = nib.Nifti1Image(y_est[:,:,:,jj], data_.affine , data_.header)
                 nib.save(data_,out_path + diff_maps[jj])
        
        
        
    
    
    
if __name__ == "__main__":
    main()
        
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
 
    
