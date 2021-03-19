# this script installs Sean Naren's warpctc loss function found here:
## https://github.com/SeanNaren/warp-ctc

# You may need to uninstall existing versions of CUDA and install CUDA 9.2

########### UNINSTALLING CUDA ###########
# ## documentation on uninstalling cuda
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#handle-uninstallation

# ## Use the following commands to uninstall a RPM/Deb installation:
# sudo apt-get --purge remove cuda

# ## Use the following command to uninstall a Toolkit runfile installation:
# sudo /usr/local/cuda-X.Y/bin/uninstall_cuda_X.Y.pl

# ## Use the following command to uninstall a Driver runfile installation:
# sudo /usr/bin/nvidia-uninstall

# ## to see list of installed nvidia drivers
# apt list --installed | grep nvidia

# ## to uninstall the drivers
# sudo apt-get remove --purge nvidia-440 nvidia-modprobe nvidia-settings
#############################


########### INSTALLING CUDA 9.2  #############

# # cuda 9.2
# cd ~
# wget https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers/cuda_9.2.148_396.37_linux
# wget https://developer.nvidia.com/compute/cuda/9.2/Prod2/patches/1/cuda_9.2.148.1_linux
# sudo sh cuda_9.2.148_396.37_linux
# rm cuda_9.2.148_396.37_linux
# sudo sh cuda_9.2.148.1_linux
# rm cuda_9.2.148.1_linux


# follow instructions here for help: https://www.pugetsystems.com/labs/hpc/How-to-install-CUDA-9-2-on-Ubuntu-18-04-1184/
# ####################################


cd libs
git clone https://github.com/SeanNaren/warp-ctc.git warp-ctc-naren

# build the warp-ctc library
cd warp-ctc-naren
mkdir build; cd build
cmake ..
make

# install the pytorch bindings
cd ../pytorch_binding
python setup.py install

# reinstall pytorch with Cuda9.2 version, if necessary

# if an import error for `_warp_ctc.cpython-36m-x86_64-linux-gnu.so` is encoutered,
## run the commented code below
# cp ~/miniconda3/envs/pyt_16/lib/python3.6/site-packages/warpctc_pytorch-0.1-py3.6-linux-x86_64.egg/warpctc_pytorch/_warp_ctc.cpython-36m-x86_64-linux-gnu.so ~/awni_speech/speech-lfs/libs/warp-ctc-naren/pytorch_binding/warpctc_pytorch
## I don't think you need to manually install warpctc_pytorch; it may actually cause an issue.
# pip install warpctc_pytorch


# update the `LD_LIBRARY_PATH` variable
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/libs/warp-ctc-naren/build
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/libs/warp-ctc-naren/build' >> ~/awni_speech/speech-lfs/setup.sh


