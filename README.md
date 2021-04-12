# PytorchRL-Getting-Started
PyTorch - Getting Started with RL 

#Getting started on a fresh linux install

starting with Ubuntu 18.04:
Install the most up to date nvidia drivers
    The drivers I am on are release 460. They support up to cuda 11.2. Even though pytorch supports up to 11.1 this is ok. I didn't need to do anything with the nvidia toolkit either

    https://docs.nvidia.com/cuda/archive/10.2/pdf/CUDA_Installation_Guide_Linux.pdf
    https://developer.nvidia.com/cuda-11.1.1-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal

    Check Capabilities with command 'nvidia-smi'

Install docker + nvidia docker container for GPU support
    Useful docker guide:
        https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-18-04#step-7-%E2%80%94-committing-changes-in-a-container-to-a-docker-image
    
    docker run --gpus all -it --ipc=host -v ~/MachineLearning/PytorchRL-Getting-Started/:/containerdir/ --name pytorchRL-evo ubuntu:18.04 

    https://docs.docker.com/get-docker/
    https://github.com/NVIDIA/nvidia-docker
    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

    Installing Anaconda + Python3.7
    https://docs.anaconda.com/anaconda/install/linux/
    bash Anaconda3-2020.11-Linux-x86_64.sh
    conda create -n py37 python=3.7
    conda activate py37

    Installing torch related things
    conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
    conda install -c conda-forge ffmpeg


    Finally Installing pip related things
    python -m pip install -r requirements.txt


Download and run the docker container

Found existing docker containers for pytorch with cuda, cudnn, devel
    https://hub.docker.com/r/pytorch/pytorch/tags?page=1&ordering=last_updated

This walkthrough was useful for starting up a prexisting pytorch docker container
    https://jacobsgill.es/pytorch-gpu-stack-setup

A nvidia container for pytorch
    https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_21-02.html#rel_21-02
    https://ngc.nvidia.com/catalog/containers/nvidia:pytorch



trying : pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
ok... making progress. PPO Half-Cheetah is working... SAC is not but i guess it isn't a cuda problem anymore

I didn't need to use custom drivers with cuda 11.1 support. All i needed was pytorch with the 11.1 cuda compiled. Aka just run : pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html after installing up to date Nvidia drivers worked




Important command : 
docker run --gpus all -it --ipc=host -v ~/localdir/:/containerdir/ --name mypytorchproject pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel 

docker run --gpus all -it --ipc=host -v ~/localdir/:/containerdir/ --name pytorch_test_environment ubuntu 


Issues with video recording with gym==0.18.x. Downgrading to 0.17.3 to fix the problem. See following for more details relating to this bug: 
    https://github.com/openai/gym/issues/1925
    https://github.com/openai/gym/pull/2151


#Saving Docker environments for use later... 
https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-18-04#step-7-%E2%80%94-committing-changes-in-a-container-to-a-docker-image

docker pull tbarkin/pytorch_rl
docker run --gpus all -it --ipc=host -v ~/MachineLearning/PytorchRL-Getting-Started/:/containerdir/ --name pytorchRL-evo tbarkin/pytorch_rl:latest 
