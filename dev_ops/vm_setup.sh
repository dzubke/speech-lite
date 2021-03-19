# this script installs the linux packages and python environments 


echo -e "\n=========  updating VM  =========\n"
sudo apt-get update
sudo apt-get --only-upgrade install kubectl google-cloud-sdk 
sudo apt install -y build-essential
sudo apt-get install -y manpages-dev cmake make

echo -e "\n=========  installing miniconda  =========\n"
sudo apt-get -y install bzip2  # need to install bzip2 to install miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh
bash Miniconda3-4.5.4-Linux-x86_64.sh -b -p $HOME/miniconda
rm Miniconda3-4.5.4-Linux-x86_64.sh
# enabling conda in future shells sessions
echo '# adding conda path
. /home/dzubke/miniconda/etc/profile.d/conda.sh' >> ~/.bashrc
# enabling conda in this current shell sessions
export PATH=$PATH:~/miniconda/bin/



echo  -e "\n=========  installed git  =========\n"
sudo apt-get install -y1 python-software-properties software-properties-common
sudo add-apt-repository ppa:git-core/ppa -y
sudo apt-get update
sudo apt-get install -y git
# install git-lfs
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

echo  -e "\n========= cloning repo  =========\n"
mkdir awni_speech && cd ~/awni_speech
git clone https://github.com/dzubke/speech-lfs.git


echo  -e "\n=========  creating conda env  =========\n"
conda create -y -n awni_env36 python=3.6.5
conda activate awni_env36
pip install --upgrade pip

echo  -e "\n=========  installing python requirements  =========\n"
cd ~/awni_speech/speech-lfs
while read requirement; do conda install --yes $requirement || pip install $requirement; done < requirements.txt

# 10. install CUDA
    # before adding GPU to VM install the driver - Tesla K80 can use CUDA 9.0, 9.2, 10.0, 10.1, and maybe others
    # location of CUDA archive - http://developer.download.nvidia.com/compute/cuda/repos/
    # source for below: https://developer.nvidia.com/cuda-92-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=debnetwork & https://cloud.google.com/compute/docs/gpus/add-gpus#install-gpu-driver

echo  -e "\n=========  installing CUDA  =========\n"
# driver website: https://www.nvidia.com/content/DriverDownload-March2009/confirmation.php?url=/tesla/396.82/NVIDIA-Linux-x86_64-396.82.run&lang=us&type=Tesla
# installer download: https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers/cuda_9.2.148_396.37_linux

# cuda 9.2
#wget https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers/cuda_9.2.148_396.37_linux
#wget https://developer.nvidia.com/compute/cuda/9.2/Prod2/patches/1/cuda_9.2.148.1_linux
#sudo sh cuda_9.2.148_396.37_linux
#sudo sh cuda_9.2.148.1_linux
# follow instructions here for help: https://www.pugetsystems.com/labs/hpc/How-to-install-CUDA-9-2-on-Ubuntu-18-04-1184/

#cuda 10.0
# curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
# sudo dpkg -i cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
# sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
# sudo apt-get -y update
# sudo apt-get -y install cuda
# rm ~/awni_speech/speech/cuda-repo-ubuntu1604_10.0.130-1_amd64.deb

# cuda 10.2
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-ubuntu1604.pin
sudo mv cuda-ubuntu1604.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/ /"
sudo apt-get update
sudo apt-get install -y cuda=10.2.89-1

# cuda 10.0
# curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
# sudo dpkg -i cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
# sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
# sudo apt-get update
# sudo apt-get install -y cuda=10.0.130-1


echo  -e "\n=========  installing linux packages  =========\n"
sudo apt-get -y install ffmpeg sox vim
echo 'set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab' >> ~/.vimrc


echo  -e "\n=========  installing pytorch  =========\n"
conda install -y pytorch=0.4.1 cuda102 -c pytorch
# pytoch 1.3 version for mac
##conda install -y pytorch torchvision -c pytorch
# using the `cudatoolkit=9.2` structure below doesn't seem to work as well as `cuda92`
## conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=9.2 -c pytorch



echo  -e "\n=========  making project libraries  =========\n"
cd ~/awni_speech/speech-lfs/
make

# build naren's loss function
cd ~/awni_speech/speech-lfs/libs/
git clone https://github.com/SeanNaren/warp-ctc.git warp-ctc-naren
cd warp-ctc-naren
mkdir build; cd build
cmake ..; make
cd pytorch_binding; python setup.py install



echo  -e "\n=========  configuring data disk  =========\n"
sudo mkdir -p /mnt/disks/data_disk
echo UUID=`sudo blkid -s UUID -o value /dev/sdb1` /mnt/disks/data_disk ext4 ro,discard,suid,dev,exec,auto,nouser,async,nofail,noload 0 2 | sudo tee -a /etc/fstab

mkdir logs


echo  -e "\n=========  configuring ~/.bashrc  =========\n"
echo '# setup
conda activate awni_env36
cd ~/awni_speech/speech-lfs
source setup.sh

#aliases
alias ..="cd .."

alias ..2="cd ../.."

alias ..3="cd ../../.."

alias rl="readlink -f"

alias watchc="watch -d -n 0.3"

TZ=America/Denver; export TZ
' >> ~/.bashrc



echo  -e "\n=======  setup complete  =========\n"
