#!/bin/sh
#SBATCH --time=10:00:00          # Run time in hh:mm:ss
#SBATCH --mem=50000              # Maximum memory required (in megabytes)
#SBATCH --job-name=idkTest_new
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --nodes=1
#SBATCH --output=/work/netthinker/ayush/2_homework.out

module load anaconda
source activate tensorflow-gpu-1.7.0-py36
python -u $@ --log_dir="/work/netthinker/ayush/new_logs/idkTest/" --model_dir="/work/netthinker/ayush/new_hw_1_logs/idkTest/" --numOfLayers=4 --batch_size=100 --epochs=950 --learning_rate=0.0001 --momentum_1=0.99 --momentum_2=0.999 --reg_scale=0.0
