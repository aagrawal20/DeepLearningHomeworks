#!/bin/sh
#SBATCH --time=168:00:00          # Run time in hh:mm:ss
#SBATCH --mem=50000              # Maximum memory required (in megabytes)
#SBATCH --job-name=agent_2
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --nodes=1
#SBATCH --output=/work/cse496dl/teams/Dropouts/3_Homework/out_files/agent_2.out

module load anaconda
source activate rl_tensorflow
python -u $@ --model_dir="/work/cse496dl/teams/Dropouts/3_Homework/agent_2/" --batch_size=64 --ep_num=5000 --learning_rate=0.0003 --target_update=500
