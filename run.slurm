#!/bin/bash
#SBATCH --partition=gpuq                    # the DGX only belongs in the 'gpu'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QoS
#SBATCH --job-name=recbole-job-ranking-emcdr
#SBATCH --output=/home/dmeher/slurm_outputs/recbole-job-ranking-emcdr.%j.out
#SBATCH --error=/home/dmeher/slurm_outputs/recbole-job-ranking-emcdr.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                 # up to 128; 
#SBATCH --gres=gpu:A100.40gb:1
#SBATCH --mem-per-cpu=80GB                 # memory per CORE; total memory is 1 TB (1,000,000 MB)
#SBATCH --export=ALL 
#SBATCH --time=5-00:00:00                   # set to 1hr; please choose carefully
#SBATCH --mail-type=BEGIN,END,FAIL     # NONE,BEGIN,END,FAIL,REQUEUE,ALL,...
#SBATCH --mail-user=dmeher@gmu.edu   # Put your GMU email address here

set echo
umask 0027

# to see ID and state of GPUs assigned
nvidia-smi

module load gnu10
module load python

source /home/dmeher/llm_env/bin/activate

python /home/dmeher/Recbole-CDR-LLMasRec/run_recbole_cdr.py --config_files=/home/dmeher/Recbole-CDR-LLMasRec/config_files/ranking_config.yaml --model=EMCDR --learning_rate=0.001 --mapping_function=non_linear --mlp_hidden_size=[128] --overlap_batch_size=300 --reg_weight=0.01 --latent_factor_model=BPR --loss_type=BPR
