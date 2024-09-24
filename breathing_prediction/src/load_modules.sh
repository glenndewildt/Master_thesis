module load 2023
module load Anaconda3/2023.07-2

module load tensorboard/2.15.1-gfbf-2023a
module load librosa/0.10.1-foss-2023a
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

module load librosa/0.10.1-foss-2023a
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 

echo "Module list"
echo "+++++++++++++++++++++++++++++"
module list
 
echo "+++++++++++++++++++++++++++++"
echo "active environment: $CONDA_PREFIX"
echo "+++++++++++++++++++++++++++++"
module save "thesis"
