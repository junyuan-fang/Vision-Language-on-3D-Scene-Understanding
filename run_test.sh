#!/bin/bash
#SBATCH --job-name=clip_voxel_cls
#SBATCH --account=project_2002051
#SBATCH --partition=test
##SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
##SBATCH --gres=gpu:a100:1
## if local fast disk on a node is also needed, replace above line with:
##SBATCH --gres=gpu:a100:1,nvme:900
#
## Please remember to load the environment your application may need.
## And use the variable $LOCAL_SCRATCH in your batch job script 
## to access the local fast storage on each node.

python Training_for_ScanObjectNN_whole_visual_part_n_way_dataset.py