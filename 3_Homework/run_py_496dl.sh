#!/bin/sh
#SBATCH --time=120:00:00          # Run time in hh:mm:ss
#SBATCH --mem=25000              # Maximum memory required (in megabytes)
#SBATCH --job-name=agent_13
#SBATCH --partition=cse496dl
#SBATCH --gres=gpu
#SBATCH --nodes=1
#SBATCH --output=/work/cse496dl/teams/Dropouts/3_Homework/out_files/agent_13.out

python -u $@ --model_dir="/work/cse496dl/teams/Dropouts/3_Homework/agent_13/" --start_learning=10000 --batch_size=32 --learning_rate=0.0001 --target_update=10000 --eps_start=1.0 --eps_end=0.01 --eps_decay=10000 --max_steps_per_game=1000 --steps_to_take=100000
