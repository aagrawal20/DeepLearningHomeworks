#!/bin/sh
#SBATCH --time=72:00:00          # Run time in hh:mm:ss
#SBATCH --mem=25000              # Maximum memory required (in megabytes)
#SBATCH --job-name=agent_8
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --nodes=1
#SBATCH --output=/work/cse496dl/teams/Dropouts/3_Homework/out_files/agent_8.out

python -u $@ --model_dir="/work/cse496dl/teams/Dropouts/3_Homework/agent_8/" --max_steps=5000 --batch_size=64 --ep_num=1000 --learning_rate=0.0005 --target_update=50 --eps_start=1.0 --eps_end=0.01 --eps_decay=1000000
