#!/bin/sh
#SBATCH --time=10:00:00          # Run time in hh:mm:ss
#SBATCH --mem=50000              # Maximum memory required (in megabytes)
#SBATCH --job-name=aagrawal_atendle_homework_1
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --nodes=1
#SBATCH --output=/work/netthinker/ayush/256_TwoLayer_fixed.out

module load anaconda
source activate tensorflow-gpu-1.7.0-py36
python -u $@ --batch_size=100 --epochs=1000
