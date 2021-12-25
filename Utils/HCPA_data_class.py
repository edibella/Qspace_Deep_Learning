import json
import os
import sys

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

class HCPA_Data(object):
    """This class initiates an object that will read in the subject 
    information file and returns the appropriate data in a clean 
    fashion.

    Arguments
    ---------
    data_path: str
        The path of fully sampled data

    sampling: str
        undersampling pattern in q-space
        
        example : 
        30_DWI -> 15 AP + 15 PA dwis with asymmetric AP and PA, preprocessed independently
        
    subject_id: str (optional)
        The subject id in the study 
        If nothing is passed then it will list all of the subjects in the data_path

    """

    def __init__(self, data_path, sampling = '30_DWI', subject_id=None):
        
        self.data_path = data_path
        self.sampling = sampling
        self.subject_id = subject_id
        self.diff_map_dict = {'fa':'dti_FA.nii.gz',
                              'md':'dti_MD.nii.gz',
                              'rd':'dti_RD.nii.gz',
                              'ak':'dki_AK.nii.gz',
                              'mk':'dki_MK.nii.gz',
                              'rk':'dki_RK.nii.gz',
                              'odi':'noddi_ODI.nii.gz',
                              'fiso':'noddi_FISO.nii.gz',
                              'ficvf':'noddi_FICVF.nii.gz',
                              'v_int':'smt_v_int.nii.gz',
                              'lambda':'smt_lambda.nii.gz',
                              'mask':'mask.nii.gz',
                              'dwi':'DWI.nii.gz'}
        

        if self.subject_id == None:
            print("Choose subjects from the following list:\n")
            for s_id in os.listdir(self.data_path):
                print(f'\t{s_id}')
            sys.exit()

       
        
    

    def get_data(self):
        """
        Returns the data
    
        Returns
        -------
        data_type: dict of arrays
            The data in a dictionary for each subject for all
            metrics and full data
        """

        
        data_subject = {}
        data_subject["dwi"] = self._return_data("dwi",False)
        data_subject["mask"] = self._return_data("mask",False)
        data_subject["fa"] = self._return_data("fa",False)
        data_subject["md"] = self._return_data("md",False)
        data_subject["rd"] = self._return_data("rd",False)
        data_subject["ak"] = self._return_data("ak",False)
        data_subject["mk"] = self._return_data("mk",False)
        data_subject["rk"] = self._return_data("rk",False)
        data_subject["odi"] = self._return_data("odi",False)
        data_subject["fiso"] = self._return_data("fiso",False)
        data_subject["ficvf"] = self._return_data("ficvf",False)
        data_subject["v_int"] = self._return_data("v_int",False)
        data_subject["lambda"] = self._return_data("lambda",False)

        return data_subject   
    
            
    

    def get_dwi(self,return_nifti=False):
        """
        Returns the DWIs (sdirections) image pixel values

        Arguments
        ---------
        return_nifti: bool
            Whether or not you want the nibabel object or numpy
            array on return
        
        Returns
        -------
        data:  array_like
            The full 4D dataset (x, y, d, z)
        """

        return self._return_data("dwi", return_nifti)
    
    def get_mask(self,return_nifti=False):
        """
        Returns the mask

        Arguments
        ---------
        return_nifti: bool
            Whether or not you want the nibabel object or numpy
            array on return
        
        Returns
        -------
        data:  array_like
            The full 4D dataset (x, y, z)
        """
            
        return self._return_data("mask", return_nifti)
    
    
    def get_fa(self, return_nifti=False):
        """
        Returns the fractional anistropy map pixel values

        Arguments
        ---------
        return_nifti: bool
            Whether or not you want the nibabel object or numpy
            array on return

        Returns
        -------
        data:  array_like
            The 3D FA map (x, y, z)
        """
        return self._return_data("fa", return_nifti)
    
    
    def get_md(self, return_nifti=False):
        """
        Returns the mean diffusivity map pixel values

        Arguments
        ---------
        return_nifti: bool
            Whether or not you want the nibabel object or numpy
            array on return

        Returns
        -------
        data:  array_like
            The 3D MD map (x, y, z)
        """
        return self._return_data("md", return_nifti)
    
    
    def get_rd(self, return_nifti=False):
        """
        Returns the radial diffusivity map pixel values

        Arguments
        ---------
        return_nifti: bool
            Whether or not you want the nibabel object or numpy
            array on return

        Returns
        -------
        data:  array_like
            The 3D RD map (x, y, z)
        """
        return self._return_data("rd", return_nifti)

    def get_ak(self, return_nifti=False):
        """
        Returns the axial kurtosis map pixel values

        Arguments
        ---------
        return_nifti: bool
            Whether or not you want the nibabel object or numpy
            array on return

        Returns
        -------
        data:  array_like
            The 3D AK map (x, y, z)
        """
        return self._return_data("ak", return_nifti)
    
    
    def get_mk(self, return_nifti=False):
        """
        Returns the mean kurtosis map pixel values

        Arguments
        ---------
        return_nifti: bool
            Whether or not you want the nibabel object or numpy
            array on return

        Returns
        -------
        data:  array_like
            The 3D MK map (x, y, z)
        """
        return self._return_data("mk", return_nifti)
    
    
    def get_rk(self, return_nifti=False):
        """
        Returns the radial kurtosis map pixel values

        Arguments
        ---------
        return_nifti: bool
            Whether or not you want the nibabel object or numpy
            array on return

        Returns
        -------
        data:  array_like
            The 3D RK map (x, y, z)
        """
        return self._return_data("rk", return_nifti)

    def get_odi(self, return_nifti=False):
        """
        Returns the orientaion dispresion index map pixel values

        Arguments
        ---------
        return_nifti: bool
            Whether or not you want the nibabel object or numpy
            array on return

        Returns
        -------
        data:  array_like
            The 3D ODI map (x, y, z)
        """
        return self._return_data("odi", return_nifti)

    def get_fiso(self,return_nifti=False):
        """
        Returns the CSF volume fraction map image pixel values

        Arguments
        ---------
        return_nifti: bool
            Whether or not you want the nibabel object or numpy
            array on return

        Returns
        -------
        data:  array_like
            The 3D FISO map (x, y, z)
        """
        return self._return_data("fiso", return_nifti)


    def get_ficvf(self, return_nifti=False):
        """
        Returns the intra-cellular volume fraction map pixel values

        Arguments
        ---------
        return_nifti: bool
            Whether or not you want the nibabel object or numpy
            array on return

        Returns
        -------
        data:  array_like
            The 3D FICVF map (x, y, z)
        """
        return self._return_data("ficvf", return_nifti)

    def get_v_int(self,return_nifti=False):
        """
        Returns the intra-axonal volume fraction map image pixel values

        Arguments
        ---------
        return_nifti: bool
            Whether or not you want the nibabel object or numpy
            array on return

        Returns
        -------
        data:  array_like
            The 3D v_int map (x, y, z)
        """
        return self._return_data("v_int", return_nifti)


    def get_lambda(self, return_nifti=False):
        """
        Returns the intrinsic diffusivity map pixel values

        Arguments
        ---------
        return_nifti: bool
            Whether or not you want the nibabel object or numpy
            array on return

        Returns
        -------
        data:  array_like
            The 3D lambda map (x, y, z)
        """
        return self._return_data("lambda", return_nifti)
    
    



   
    

    def _return_data(self, img_type, return_nifti):

        if (img_type == 'dwi') or (img_type == 'mask'):
            img_path = self.data_path + self.sampling + '/processed/'+self.subject_id + '/' + self.diff_map_dict[img_type]
        else:
            img_path = self.data_path + 'Fully_Sampled/' + self.subject_id + '/' + self.diff_map_dict[img_type]

        
        img = nib.load(img_path)
        
        if return_nifti:
            return img
        else:
            img_np = img.get_data()
            return img_np

        
def main():
    import HCPA_data_class
    
    test_data_obj = HCPA_data_class.HCPA_Data('./../Data/',sampling = '30_DWI', subject_id = 'M_56_Strk')
    test_data = test_data_obj.get_data()
    print(test_data.keys())
    print(test_data['dwi'].shape)
    print(test_data['odi'].shape)
    x_crop = slice(25,125)
    y_crop = slice(1,120)
    z_slice = 35
    dwi_0 = test_data['dwi'][x_crop,y_crop,z_slice,0]
    dwi_1 = test_data['dwi'][x_crop,y_crop,z_slice,1]
    dwi_2 = test_data['dwi'][x_crop,y_crop,z_slice,2]
    dwi_3 = test_data['dwi'][x_crop,y_crop,z_slice,3]
    dwi_4 = test_data['dwi'][x_crop,y_crop,z_slice,4]
    dwi_5 = test_data['dwi'][x_crop,y_crop,z_slice,5]
    dwi_6 = test_data['dwi'][x_crop,y_crop,z_slice,6]
    dwi_7 = test_data['dwi'][x_crop,y_crop,z_slice,7]
    dwi_8 = test_data['dwi'][x_crop,y_crop,z_slice,8]
    dwi_9 = test_data['dwi'][x_crop,y_crop,z_slice,9]
    dwi_10 = test_data['dwi'][x_crop,y_crop,z_slice,10]
    dwi_11 = test_data['dwi'][x_crop,y_crop,z_slice,11]
    fa = test_data['fa'][x_crop,y_crop,z_slice]
    md = test_data['md'][x_crop,y_crop,z_slice]
    rd = test_data['rd'][x_crop,y_crop,z_slice]
    ak = test_data['ak'][x_crop,y_crop,z_slice]
    mk = test_data['mk'][x_crop,y_crop,z_slice]
    rk = test_data['rk'][x_crop,y_crop,z_slice]
    odi = test_data['odi'][x_crop,y_crop,z_slice]
    ficvf = test_data['ficvf'][x_crop,y_crop,z_slice]
    fiso = test_data['fiso'][x_crop,y_crop,z_slice]
    v_int = test_data['v_int'][x_crop,y_crop,z_slice]
    lambda_int = test_data['lambda'][x_crop,y_crop,z_slice]
    lambda_ext = (1-2*v_int/3)*lambda_int
    
    
    fig = plt.figure(figsize = (15*(100/130)*1.02,9*1.033))
    fig.add_subplot(3,5,1)
    plt.title("b0")
    plt.imshow(np.rot90(dwi_0), cmap = 'gray', vmin = 0, vmax = np.max(dwi_0))
    plt.axis('off')
    
    fig.add_subplot(3,5,6)
    plt.title("b = 1500")
    plt.imshow(np.rot90(dwi_1), cmap = 'gray', vmin = 0, vmax = np.max(dwi_1))
    plt.axis('off')
    
    fig.add_subplot(3,5,11)
    plt.title("b = 3000")
    plt.imshow(np.rot90(dwi_2), cmap = 'gray', vmin = 0, vmax = np.max(dwi_2))
    plt.axis('off')
    
    fig.add_subplot(3,5,2)
    plt.title("FA")
    plt.imshow(np.rot90(fa), cmap = 'gray', vmin = 0, vmax = 1)
    plt.axis('off')
    
    fig.add_subplot(3,5,7)
    plt.title("MD")
    plt.imshow(np.rot90(md), cmap = 'gray', vmin = 0, vmax = np.max(md))
    plt.axis('off')
    
    fig.add_subplot(3,5,12)
    plt.title("RD")
    plt.imshow(np.rot90(rd), cmap = 'gray', vmin = 0, vmax = np.max(rd))
    plt.axis('off')
    
    fig.add_subplot(3,5,3)
    plt.title("AK")
    plt.imshow(np.rot90(ak), cmap = 'gray', vmin = 0, vmax = np.max(ak))
    plt.axis('off')
    
    fig.add_subplot(3,5,8)
    plt.title("MK")
    plt.imshow(np.rot90(mk), cmap = 'gray', vmin = 0, vmax = np.max(mk))
    plt.axis('off')
    
    fig.add_subplot(3,5,13)
    plt.title("RK")
    plt.imshow(np.rot90(rk), cmap = 'gray', vmin = 0, vmax = np.max(rk))
    plt.axis('off')
    
    fig.add_subplot(3,5,4)
    plt.title("ODI")
    plt.imshow(np.rot90(odi), cmap = 'gray', vmin = 0, vmax = 1)
    plt.axis('off')
    
    fig.add_subplot(3,5,9)
    plt.title("FICVF")
    plt.imshow(np.rot90(ficvf), cmap = 'gray', vmin = 0, vmax = 1)
    plt.axis('off')
    
    fig.add_subplot(3,5,14)
    plt.title("FISO")
    plt.imshow(np.rot90(fiso), cmap = 'gray', vmin = 0, vmax = 1)
    plt.axis('off')
    
    fig.add_subplot(3,5,5)
    plt.title("$v_{int}$")
    plt.imshow(np.rot90(v_int), cmap = 'gray', vmin = 0, vmax = 1)
    plt.axis('off')
    
    fig.add_subplot(3,5,10)
    plt.title("$\lambda$")
    plt.imshow(np.rot90(lambda_int), cmap = 'gray', vmin = 0, vmax = np.max(lambda_int))
    plt.axis('off')
    
    fig.add_subplot(3,5,15)
    plt.title("$\lambda_{ext}$")
    plt.imshow(np.rot90(lambda_ext), cmap = 'gray', vmin = 0, vmax = np.max(lambda_ext))
    plt.axis('off')
    
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    
    
    fig_1 = plt.figure(figsize = (12*(100/130)*1.025,9*1.033))
    fig_1.add_subplot(3,4,1)
    plt.imshow(np.rot90(dwi_0), cmap = 'gray', vmin = 0, vmax = np.max(dwi_0))
    plt.axis('off')
    
    fig_1.add_subplot(3,4,2)
    plt.imshow(np.rot90(dwi_1), cmap = 'gray', vmin = 0, vmax = np.max(dwi_1))
    plt.axis('off')
    
    fig_1.add_subplot(3,4,3)
    plt.imshow(np.rot90(dwi_2), cmap = 'gray', vmin = 0, vmax = np.max(dwi_2))
    plt.axis('off')
    
    fig_1.add_subplot(3,4,4)
    plt.imshow(np.rot90(dwi_3), cmap = 'gray', vmin = 0, vmax = np.max(dwi_3))
    plt.axis('off')
    
    fig_1.add_subplot(3,4,5)
    plt.imshow(np.rot90(dwi_4), cmap = 'gray', vmin = 0, vmax = np.max(dwi_4))
    plt.axis('off')
    
    fig_1.add_subplot(3,4,6)
    plt.imshow(np.rot90(dwi_5), cmap = 'gray', vmin = 0, vmax = np.max(dwi_5))
    plt.axis('off')
    
    fig_1.add_subplot(3,4,7)
    plt.imshow(np.rot90(dwi_6), cmap = 'gray', vmin = 0, vmax = np.max(dwi_6))
    plt.axis('off')
    
    fig_1.add_subplot(3,4,8)
    plt.imshow(np.rot90(dwi_7), cmap = 'gray', vmin = 0, vmax = np.max(dwi_7))
    plt.axis('off')
    
    fig_1.add_subplot(3,4,9)
    plt.imshow(np.rot90(dwi_8), cmap = 'gray', vmin = 0, vmax = np.max(dwi_8))
    plt.axis('off')
    
    fig_1.add_subplot(3,4,10)
    plt.imshow(np.rot90(dwi_9), cmap = 'gray', vmin = 0, vmax = np.max(dwi_9))
    plt.axis('off')
    
    fig_1.add_subplot(3,4,11)
    plt.imshow(np.rot90(dwi_10), cmap = 'gray', vmin = 0, vmax = np.max(dwi_10))
    plt.axis('off')
    
    fig_1.add_subplot(3,4,12)
    plt.imshow(np.rot90(dwi_11), cmap = 'gray', vmin = 0, vmax = np.max(dwi_11))
    plt.axis('off')
    
    fig_1.subplots_adjust(wspace=0.1, hspace=0.1)
    
    
    plt.show()


if __name__ == "__main__":
    main()

                  
        
