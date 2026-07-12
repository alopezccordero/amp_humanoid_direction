#!/bin/bash
#SBATCH --job-name=humanoid_amp_notfixed
#SBATCH --partition=gpua30q
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=amp_humanoid_dir_%j.out
#SBATCH --error=amp_humanoid_dir_%j.err

source ~/miniconda3/bin/activate
conda activate mujoco-env

# One BLAS/OpenMP thread per process. Each SubprocVecEnv worker runs torch,
# which otherwise spawns threads for every core on the node and the 8 workers
# thrash the 8-CPU allocation.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd ~/amp_humanoid_direction

echo "Running on $(hostname)"
echo "current directory: $(pwd)"
python --version

nvidia-smi

python train_sb3_real_amp.py

