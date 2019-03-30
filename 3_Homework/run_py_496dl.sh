#!/bin/sh
#SBATCH --time=24:00:00          # Run time in hh:mm:ss
#SBATCH --mem=25000              # Maximum memory required (in megabytes)
#SBATCH --job-name=agent_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --nodes=1
#SBATCH --output=/work/cse496dl/teams/Dropouts/3_Homework/out_files/agent_test.out

python -u $@ --model_dir="/work/cse496dl/teams/Dropouts/3_Homework/test_agent/" --batch_size=64 --ep_num=100 --learning_rate=0.0005 --target_update=100 --eps_start=1.0 --eps_end=0.1 --eps_decay=100000
