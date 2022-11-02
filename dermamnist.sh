#!/bin/bash

#SBATCH --gres=gpu:1 --constraint=gpu1080|gpup100
#SBATCH --job-name="GA-NAS"
#SBATCH -p publicgpu


module rm compilers/intel17
module load python/Anaconda
module rm compilers/intel17
module load compilers/cuda-10.0
module rm compilers/intel17
source activate deeplearning
module rm compilers/intel17

python dermamnist.py

