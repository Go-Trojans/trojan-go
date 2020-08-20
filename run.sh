#!/bin/bash

# Check and set the CUDA 10.1
ls -l /usr/local/cuda
sudo rm -rf /usr/local/cuda
sudo ln -s /usr/local/cuda-10.1 /usr/local/cuda

# Create the project by pulling it from git.
mkdir TROJANGO; cd TROJANGO
git clone https://CSCI561-Go:TrojanGousc123@github.com/Go-Trojans/trojan-go.git --branch aws_trojango
cd

# Set the conda environment.
conda create --name awsgo --file TROJANGO/trojan-go/spec-file.txt
source activate awsgo
pip install tensorflow-gpu==2.2

# Run the code.
cd TROJANGO/trojan-go/code/
for a in /sys/bus/pci/devices/*; do echo 0 | sudo tee -a $a/numa_node; done
