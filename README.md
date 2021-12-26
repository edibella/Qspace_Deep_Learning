# Q-space Deep Learning

https://user-images.githubusercontent.com/1512443/147379103-f3d80ff6-dcac-440b-b829-2111cfb685ae.mov

This repository provides implementation codes for "Jointly estimating parametric maps of multiple diffusion models from undersampled q-space data: A comparison of three deep learning approaches."

## Setting up a Python env and installing required packages

To run Python codes provided in this repository, create a Python environment:
```
$ conda create -p /path_to_env/env_name python=3.x
```
and install the follwoing packages
```
$ conda install -p /path_to_env/env_name/ -c anaconda numpy
$ conda install -p /path_to_env/env_name/ -c conda-forge matplotlib
$ conda install -p /path_to_env/env_name/ -c conda-forge nibabel
$ conda install -p /path_to_env/env_name/ -c anaconda h5py
$ conda install -p /path_to_env/env_name/ -c conda-forge tqdm
$ conda install -p /path_to_env/env_name/ -c conda-forge argparse
$ conda install -p /path_to_env/env_name/ pytorch torchvision torchaudio cudatoolkit=x.x -c pytorch 
```
where ``` cudatoolkit=x.x``` in the last command depends on the installed cude version, e. g. 10.2 or 11.3. After these activate your environment using:
```
$ conda activate /path_to_env/env_name
```
## Training and Testing the 1D-qDL network

![q-DL-new](https://user-images.githubusercontent.com/1512443/147395592-db7c2997-bc43-4247-8b24-fcd6b938a4f3.png)

The 1D-qDL, as shown in the above block diagram, uese fully connected layers to jointly estimate parametric diffusion maps on a per-voxel basis. The implementation codes are under ```./1D-qDL/```. The training data are not provided in this repository, but one test dataset, trained models for three undersampling patterns, and the test results can be found at ```./Data/```.

Training:
```
usage: python train_1d.py [-h] [--sampling SAMPLING] [--batch_size BATCH_SIZE]
                   [--num_of_channels NUM_OF_CHANNELS]
                   [--num_of_layers NUM_OF_LAYERS]
                   [--num_of_hidden NUM_OF_HIDDEN] [--drop_out DROP_OUT]
                   [--epochs EPOCHS] [--lr LR] [--data_path DATA_PATH]

Training 1D-qDL

optional arguments:
  -h, --help            show this help message and exit
  --sampling SAMPLING   Q-space undersampling pattern name
  --batch_size BATCH_SIZE
                        Training batch size
  --num_of_channels NUM_OF_CHANNELS
                        Number of qDL input channels
  --num_of_layers NUM_OF_LAYERS
                        Number of qDL layers
  --num_of_hidden NUM_OF_HIDDEN
                        Number of hidden nodes in qDL
  --drop_out DROP_OUT   Drop out probability in qDL
  --epochs EPOCHS       Number of training epochs
  --lr LR               Initial learning rate
  --data_path DATA_PATH
                        Path to the data
 
 ```
 Testing: 
 
 ```
 usage: python test_1d.py [-h] [--sampling SAMPLING] [--batch_size BATCH_SIZE]
                  [--num_of_channels NUM_OF_CHANNELS]
                  [--num_of_layers NUM_OF_LAYERS]
                  [--num_of_hidden NUM_OF_HIDDEN] [--drop_out DROP_OUT]
                  [--data_path DATA_PATH] [--test_cases TEST_CASES]

1D-qDL testing

optional arguments:
  -h, --help            show this help message and exit
  --sampling SAMPLING   Q-space undersampling pattern name
  --batch_size BATCH_SIZE
                        Testing batch size
  --num_of_channels NUM_OF_CHANNELS
                        Number of qDL input channels
  --num_of_layers NUM_OF_LAYERS
                        Number of qDL layers
  --num_of_hidden NUM_OF_HIDDEN
                        Number of hidden nodes in qDL
  --drop_out DROP_OUT   Drop out probability in qDL
  --data_path DATA_PATH
                        Path to the data
  --test_cases TEST_CASES
                        List of test subjects ids 
 ```
 
 ## Training and testing the 2D-CNN network
 
 ![2D-CNN-New](https://user-images.githubusercontent.com/1512443/147397147-f2892f57-5164-40a4-8e4b-cf562e2a6311.png)

The 2D-CNN uses convolutional blocks with residual connections to jointly estimate diffusion parametric maps on a per-slice basis. The implementation codes are under ```./2D_CNN/```.
 
 Training:
 ```
 usage: python train_2d.py [-h] [--sampling SAMPLING] [--batch_size BATCH_SIZE]
                   [--num_of_channels NUM_OF_CHANNELS] [--epochs EPOCHS]
                   [--lr LR] [--data_path DATA_PATH]

Training 2D-CNN

optional arguments:
  -h, --help            show this help message and exit
  --sampling SAMPLING   Q-space undersampling pattern name
  --batch_size BATCH_SIZE
                        Training batch size
  --num_of_channels NUM_OF_CHANNELS
                        Number of CNN input channels
  --epochs EPOCHS       Number of training epochs
  --lr LR               Initial learning rate
  --data_path DATA_PATH
                        Path to the data 
 ```
 Testing:
 
 ```
 usage: python test_2d.py [-h] [--sampling SAMPLING] [--batch_size BATCH_SIZE]
                  [--num_of_channels NUM_OF_CHANNELS] [--data_path DATA_PATH]
                  [--test_cases TEST_CASES]

2D-CNN testing

optional arguments:
  -h, --help            show this help message and exit
  --sampling SAMPLING   Q-space undersampling pattern name
  --batch_size BATCH_SIZE
                        Testing batch size
  --num_of_channels NUM_OF_CHANNELS
                        Number of CNN input channels
  --data_path DATA_PATH
                        Path to the data
  --test_cases TEST_CASES
                        List of test subjects ids 
 ```
 
 ## Training and Testing the MESC-SD network
 
 ![MESC-SD](https://user-images.githubusercontent.com/1512443/147397309-c39685ab-a23e-4e22-af4b-a5198bb1bbf8.png)

The MESC-SD uses a dictionary-based sparse coding representation to jointly estimate parametric diffusion maps using 3D input patches. The implementation codes are under ```./3D_MESC_SD/```.
 
 Training:
 
 ```
 usage: python train_3d.py [-h] [--sampling SAMPLING] [--batch_size BATCH_SIZE]
                   [--num_of_channels NUM_OF_CHANNELS]
                   [--num_of_voxels NUM_OF_VOXELS]
                   [--num_hidden_lstm NUM_HIDDEN_LSTM]
                   [--num_hidden_fc NUM_HIDDEN_FC] [--epochs EPOCHS] [--lr LR]
                   [--data_path DATA_PATH]

Training MESC-SD

optional arguments:
  -h, --help            show this help message and exit
  --sampling SAMPLING   Q-space undersampling pattern name
  --batch_size BATCH_SIZE
                        Training batch size
  --num_of_channels NUM_OF_CHANNELS
                        Number of MESC_SD input channels
  --num_of_voxels NUM_OF_VOXELS
                        Number of voxels in 3D input patches
  --num_hidden_lstm NUM_HIDDEN_LSTM
                        Number of hidden nodes in LSTM units
  --num_hidden_fc NUM_HIDDEN_FC
                        Number of hidden nodes in FC layers
  --epochs EPOCHS       Number of training epochs
  --lr LR               Initial learning rate
  --data_path DATA_PATH
                        Path to the data 
 ```
 
 Testing:
 
 ```
 usage: python test_3d.py [-h] [--sampling SAMPLING] [--batch_size BATCH_SIZE]
                  [--num_of_channels NUM_OF_CHANNELS]
                  [--num_of_voxels NUM_OF_VOXELS]
                  [--num_hidden_lstm NUM_HIDDEN_LSTM]
                  [--num_hidden_fc NUM_HIDDEN_FC] [--data_path DATA_PATH]
                  [--test_cases TEST_CASES]

MESC-SD testing

optional arguments:
  -h, --help            show this help message and exit
  --sampling SAMPLING   Q-space undersampling pattern name
  --batch_size BATCH_SIZE
                        Testing batch size
  --num_of_channels NUM_OF_CHANNELS
                        Number of MESC_SD input channels
  --num_of_voxels NUM_OF_VOXELS
                        Number of voxels in 3D input patches
  --num_hidden_lstm NUM_HIDDEN_LSTM
                        Number of hidden nodes in LSTM units
  --num_hidden_fc NUM_HIDDEN_FC
                        Number of hidden nodes in FC layers
  --data_path DATA_PATH
                        Path to the data
  --test_cases TEST_CASES
                        List of test subjects ids`
 
 ```
 
