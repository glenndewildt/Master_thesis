#!/bin/bash --login

 
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
 
# includes useful libraries/applications such as ffmpeg
# Initialize conda shell
conda init bash
 
conda activate base
module load 2022
 
module load scikit-build/0.11.1-foss-2022a
module load SciPy-bundle/2022.05-foss-2022a
module load FFmpeg/4.4.2-GCCcore-11.3.0
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0


# #module load cuDNN/8.6.0.163-CUDA-11.8.0
# !module load TensorFlow/2.13.0-foss-2023a
# #!module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
#module load matplotlib/3.5.2-foss-2022a
module load 2023
module load tensorboard/2.15.1-gfbf-2023a

module load librosa/0.10.1-foss-2023a
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 
pip install audiomentations

echo "Module list"
echo "+++++++++++++++++++++++++++++"
module list
 
echo "+++++++++++++++++++++++++++++"
echo "active environment: $CONDA_PREFIX"
echo "+++++++++++++++++++++++++++++"


# Execute program located in $HOME
srun ipython $HOME/ondemand/Breathing_classification/train_wav2vec.py
