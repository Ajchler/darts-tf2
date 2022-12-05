#!/usr/bin/env bash
#PBS -q qgpu
#PBS -N d4cbatch2
#PBS -l select=1:ngpus=1,walltime=12:00:00
#PBS -A OPEN-20-37

cd $HOME/darts4coatnet

ml cuDNN/8.2.2.26-CUDA-11.4.1  Anaconda3 
source activate darts

date 2>&1
python3 search_cell.py --batch_size 128

python3 train.py --genotype_file genotypes/genotype_epoch_50 --batch_size 128 --epochs 600 --init_channels 36 --layers 20 --cutout --auxiliary
