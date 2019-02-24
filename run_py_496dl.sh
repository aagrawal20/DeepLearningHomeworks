#!/bin/sh
#SBATCH --time=60:00:00          # Run time in hh:mm:ss
#SBATCH --mem=50000              # Maximum memory required (in megabytes)
#SBATCH --job-name=simple_conv
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --nodes=1
#SBATCH --output=/work/cse496dl/atendle/1_auto.out

module load tensorflow-gpu/py36/1.7
python -u $@ --model_dir="/work/cse496dl/atendle/hw_2_logs/1_auto/" --load_data='ImageNet' --batch_size=100 --epochs=1000 --learning_rate=0.001 --momentum_1=0.99 --momentum_2=0.999 --filter_1=32 --filter_2=64 --kernel_size=5
