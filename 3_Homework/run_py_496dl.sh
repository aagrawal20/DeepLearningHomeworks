#!/bin/sh
#SBATCH --time=24:00:00          # Run time in hh:mm:ss
#SBATCH --mem=25000              # Maximum memory required (in megabytes)
#SBATCH --job-name=agent_11
#SBATCH --partition=cse496dl
#SBATCH --gres=gpu
#SBATCH --nodes=1
#SBATCH --output=/work/cse496dl/teams/Dropouts/3_Homework/out_files/agent_11.out

python -u $@ --model_dir="/work/cse496dl/teams/Dropouts/3_Homework/agent_11/" --batch_size=64 --ep_num=5000 --learning_rate=0.0005 --target_update=10 --eps_start=1.0 --eps_end=0.01 --eps_decay=100000
