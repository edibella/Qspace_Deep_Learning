
# Implementation of 1D-qDL testing
# For the github version just one test dataset is provided

import os
import sys
import h5py
import argparse
import json

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn


# importing HCPA data class and qDL model
sys.path.append("./..")
from Utils.HCPA_data_class import HCPA_Data
from model_1d import qDL

# some useful functions
def RMSE(a1,a2):
    return np.sqrt(np.sum((a1-a2)**2))

def nRMSE(a1,a_ref):
    n_ch = a_ref.shape[1]
    nrmse = np.zeros([1,n_ch])
    
    for n_c in range(n_ch):
        nrmse[0,n_c] = 2*RMSE(a_ref[:,n_c],a1[:,n_c])/RMSE(a_ref[:,n_c],-a_ref[:,n_c])
    return nrmse



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

parser = argparse.ArgumentParser(description="1D-qDL testing")
parser.add_argument("--sampling", type=str, default='30_DWI', help='Q-space undersampling pattern name')
parser.add_argument("--batch_size", type=int, default=10000, help="Testing batch size")
parser.add_argument("--num_of_channels", type=int, default=30, help="Number of qDL input channels")
parser.add_argument("--num_of_layers", type=int, default=3, help="Number of qDL layers")
parser.add_argument("--num_of_hidden", type=int, default=150, help="Number of hidden nodes in qDL")
parser.add_argument("--drop_out", type=float, default=0.1, help="Drop out probability in qDL")
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
        train_path = opt.data_path + opt.sampling +  '/train_1d/'
        x_maxs = np.array(h5py.File(train_path+'x_train_maxs.h5','r')['x_maxs'])
        x_means = np.array(h5py.File(train_path+'x_train_means.h5','r')['x_means'])
        y_means = np.array(h5py.File(train_path+'y_train_means.h5','r')['y_means'])

        
    
        print(f'Loading {subject_id} data with {opt.sampling} sampling ...\n')
        hcpa_data = HCPA_Data(opt.data_path, sampling=opt.sampling, subject_id=subject_id)
        data_dwi = hcpa_data.get_dwi()
        data_fa = hcpa_data.get_fa()
        data_md = hcpa_data.get_md()
        data_rd = hcpa_data.get_rd()
        data_ak = hcpa_data.get_ak()
        data_mk = hcpa_data.get_mk()
        data_rk = hcpa_data.get_rk()
        data_odi = hcpa_data.get_odi()
        data_fiso = hcpa_data.get_fiso()
        data_ficvf = hcpa_data.get_ficvf()
        data_v_int = hcpa_data.get_v_int()
        data_lambda = hcpa_data.get_lambda()
        data_mask = hcpa_data.get_mask()
        
        dim0, dim1, n_slices, n_channels = data_dwi.shape
        
        
        
        y_fa = data_fa.reshape(-1,1)
        y_md = data_md.reshape(-1,1)
        y_rd = data_rd.reshape(-1,1)
        y_ak = data_ak.reshape(-1,1)
        y_mk = data_mk.reshape(-1,1)
        y_rk = data_rk.reshape(-1,1)
        y_odi = data_odi.reshape(-1,1)
        y_fiso = data_fiso.reshape(-1,1)
        y_ficvf = data_ficvf.reshape(-1,1)
        y_v_int = data_v_int.reshape(-1,1)
        y_lambda = data_lambda.reshape(-1,1)
        x_mask = data_mask.reshape(-1,1)
        x_dwi = data_dwi.transpose(3,0,1,2)
        x_dwi = x_dwi.reshape(n_channels,-1)
        x_dwi = x_dwi.transpose(1,0)
        x_dwi -= x_means
        x_dwi /= x_maxs
        
     
        y_tg = np.concatenate((y_fa, y_md, y_rd, y_ak, y_mk, y_rk, y_odi, y_fiso, y_ficvf, y_v_int, y_lambda), axis=1)
        y_tg /= np.exp(np.floor(np.log(y_means)))
        samples = np.flatnonzero(x_mask[:,0] > 0)
    
        # Loading the model
        print('Loading model ...\n')
        model_path = opt.data_path + opt.sampling +  '/Models/'
        net = qDL(n_channels = opt.num_of_channels, n_out = 11, n_layers=opt.num_of_layers, n_hidden = opt.num_of_hidden, drop_out = opt.drop_out)
        device_ids = [0,1,2,3]
        b_size = len(device_ids) * opt.batch_size
        
        model = nn.DataParallel(net, device_ids=device_ids).cuda()
        model.load_state_dict(torch.load(model_path + '1D_qDL.pth'))
        model.eval()
    
    
   
        # Processing the test data
        print('Processing the test data ...\n')
    
        iterations = len(samples) // b_size
        last_batch_size = len(samples) % b_size
    
        nrmse_list = []
        y_est = np.zeros(y_tg.shape)
    
        with torch.no_grad():

            for ii in range(iterations):
                x = x_dwi[samples[ii*b_size:(ii+1)*b_size],:]
                x = torch.FloatTensor(x)
                y_target = y_tg[samples[ii*b_size:(ii+1)*b_size],:]
                y_target = torch.FloatTensor(y_target)
            
            
                x = x.cuda()
                y_target = y_target.cuda()
                y = model(x)
        
                y = y.cpu().detach().numpy()
                y_target = y_target.cpu().detach().numpy()
        
                nrmse_list.append(nRMSE(y,y_target))
                
                y_est[samples[ii*b_size:(ii+1)*b_size],:] = y

    
  
            x = x_dwi[samples[iterations*b_size:iterations*b_size+last_batch_size],:]
            x = torch.FloatTensor(x)
            y_target = y_tg[samples[iterations*b_size:iterations*b_size+last_batch_size],:]
            y_target = torch.FloatTensor(y_target)
        
        
            x = x.cuda()
            y_target = y_target.cuda()
            y = model(x)
        
            y = y.cpu().detach().numpy()
            y_target = y_target.cpu().detach().numpy()
         
            nrmse_list.append(nRMSE(y,y_target))
            
            y_est[samples[iterations*b_size:iterations*b_size+last_batch_size],:] = y
            y_est *= np.exp(np.floor(np.log(y_means)))
    
            
            nrmse_test = np.mean(np.concatenate(nrmse_list, axis = 0),axis=0)
            print('Testing Accuracy in nRMSE: ' + str(nrmse_test))
            print()
        
            # Save testing results
        
            out_path = opt.data_path + opt.sampling +  '/test_1d/' + subject_id + '/'
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            for jj in range(len(diff_maps)):
                 get_method = getattr(hcpa_data, 'get_'+ diff_methods[jj])
                 data_ = get_method(return_nifti=True)
                 data_ = nib.Nifti1Image(y_est[:,jj].reshape(dim0,dim1,n_slices), data_.affine , data_.header)
                 nib.save(data_,out_path + diff_maps[jj])
                 
                
                            
    
    
if __name__ == "__main__":
    main()
        
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
 
    
