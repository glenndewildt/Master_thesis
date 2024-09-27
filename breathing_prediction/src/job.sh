#!/bin/bash --login

 
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
 
# includes useful libraries/applications such as ffmpeg
# Initialize conda shell
module load 2023
# module load thesis

module load Anaconda3/2023.07-2

# module load tensorboard/2.15.1-gfbf-2023a
# module load librosa/0.10.1-foss-2023a
# module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# module load librosa/0.10.1-foss-2023a
# module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 

conda activate thesis

echo conda list

echo "Module list"
echo "+++++++++++++++++++++++++++++"
module list
 
echo "+++++++++++++++++++++++++++++"
echo "active environment: $CONDA_PREFIX"
echo "+++++++++++++++++++++++++++++"




# Execute program located in $HOME
srun ipython $HOME/Master_thesis/breathing_prediction/src/breathing_model_trainer.py
