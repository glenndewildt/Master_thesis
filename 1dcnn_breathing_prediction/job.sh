#!/bin/bash --login

 
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
 
# includes useful libraries/applications such as ffmpeg
# Initialize conda shell
module load 2023
#module load thesis
module load Anaconda3/2023.07-2  # Make sure Conda is loaded
#module load tensorboard/2.15.1-gfbf-2023a
#module load librosa/0.10.1-foss-2023a
#module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# module load librosa/0.10.1-foss-2023a
# module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 
source $(conda info --base)/etc/profile.d/conda.sh
#conda activate thesis
#conda init
#mamba list
conda activate thesis
conda list
conda env list
echo "Module list"
echo "+++++++++++++++++++++++++++++"
module list
 
echo "+++++++++++++++++++++++++++++"
echo "active environment: $CONDA_PREFIX"
echo "+++++++++++++++++++++++++++++"




# Execute program located in $HOME
srun python $HOME/Master_thesis/1dcnn_breathing_prediction/pt_train_bert.py
