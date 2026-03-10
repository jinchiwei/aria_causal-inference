#!/bin/bash
#SBATCH --job-name=aria-causal-dr
#SBATCH --output=/data/rauschecker2/jkw/aria/logs/aria_causal_dr_%j.out
#SBATCH --error=/data/rauschecker2/jkw/aria/logs/aria_causal_dr_%j.err
#SBATCH --time=0-04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=long

cd /data/rauschecker2/jkw/aria/dev/src

source /home/jiwei/miniconda3/bin/activate /data/rauschecker1/jkw/envs/praria

CONFIG="$1"

echo "Config: $CONFIG"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "Date: $(date)"

python run_causal_dr.py --config "$CONFIG"

echo "Completed: $CONFIG"
echo "End time: $(date)"
