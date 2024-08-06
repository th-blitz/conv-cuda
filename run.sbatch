#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:1
#SBATCH --time=08:00:00
#SBATCH --mem=32GB
#SBATCH --output=run.out

#SBATCH --mail-type=END

module purge

SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda10.2-cudnn7-devel-ubuntu18.04.sif

singularity exec --nv $SINGULARITY_IMAGE /bin/bash -c "bash run.batch"

