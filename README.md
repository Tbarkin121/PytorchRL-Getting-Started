# PytorchRL-Getting-Started
PyTorch - Getting Started with RL 

##Getting started fresh on linux
##Install python3 and pip
##Setting up the requirements.txt

##Cuda
So I think I’m trying to install cuda 11.1
It looks like 11.2 was installed with my GPU drivers? 
https://docs.nvidia.com/cuda/archive/10.2/pdf/CUDA_Installation_Guide_Linux.pdf
https://developer.nvidia.com/cuda-11.1.1-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal

Check which version of cuda is installed: 
nvidia-smi

Downgrading cuda from 11.2 to 11.1 (Nope)
https://medium.com/@praveenkrishna/downgrade-cuda-for-tensorflow-gpu-17831db59099

Trying to install Nivida GPU drivers *450
    Check which cuda is installed with this or try installing the toolkit.... 

It might be worth just going into the container environments now though... 

Ok.... uninstalled nvidia drivers
trying to install the toolkit + driver locally ...

ok, i have 11.1 now... finally
except pytorch was compiled with 10.2... checked with torch.version.cuda
trying : pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
ok... making progress. PPO Half-Cheetah is working... SAC is not but i guess it isn't a cuda problem anymore

I didn't need to use custom drivers with cuda 11.1 support. All i needed was pytorch with the 11.1 cuda compiled. Aka just run : pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html after installing up to date Nvidia drivers worked

Ok... lets look at some docker stuff next I suppose
https://docs.docker.com/get-docker/
https://github.com/NVIDIA/nvidia-docker
    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

Found existing docker containers for pytorch with cuda, cudnn, devel
    https://hub.docker.com/r/pytorch/pytorch/tags?page=1&ordering=last_updated

This walkthrough was useful for starting up a prexisting pytorch docker container
    https://jacobsgill.es/pytorch-gpu-stack-setup


A nvidia container for pytorch
    https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_21-02.html#rel_21-02
    https://ngc.nvidia.com/catalog/containers/nvidia:pytorch


Starting from a fresh ubuntu vs starting from a premade pytorch branch...

1) 
    *Installing Anaconda
    *Installing Python 3.7

Important command : 
docker run --gpus all -it --ipc=host -v ~/localdir/:/containerdir/ --name mypytorchproject pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel 

docker run --gpus all -it --ipc=host -v ~/localdir/:/containerdir/ --name pytorch_test_environment ubuntu 

Install Anaconda
https://docs.anaconda.com/anaconda/install/verify-install/
Install Python 3.7
conda create -n py37 python=3.7

trying : 
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge

Training is working from the docker environment! 
Ok, now we just have to tie this all together and push it to git... yay

Still some more clean up with the PPO
video was not recording, problem with ffmpeg
trying : 
sudo apt update
sudo apt install ffmpeg
still doesn't work... 
locally it 'records a video' but that video is not playable it seems... hmmm. Annoying... 
conda install -c conda-forge ffmpeg 
ok... they both make unplayable videos!! YAY? 
