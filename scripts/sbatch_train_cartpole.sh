#!/bin/bash
# Generic SLURM wrapper for one SB3 cartpole training arm on the iLab cluster.
#   sbatch scripts/sbatch_train_cartpole.sh <env_id> <overrides.yaml> <logfile>
# Runs from the shared-FS checkout; conda env arcmg is on /common/users, so it
# resolves on every iLab node without a sync.
#SBATCH -p unlimited
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH -o /dev/null

set -u
ENV_ID=$1
OVERRIDES=$2
LOGFILE=$3
SEED=${4:-}

source /common/users/dm1487/envs/arcmg/etc/conda/activate.d/* 2>/dev/null || true
export PATH=/common/users/dm1487/envs/arcmg/bin:$PATH
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

cd /common/home/dm1487/robotics_research/tripods/safe-control-gym
exec python3 -m safe_control_gym.experiments.train_sb3 \
    --env_id "$ENV_ID" --algo sac --output_dir logs --use_gpu \
    --overrides "$OVERRIDES" ${SEED:+--seed "$SEED"} > "$LOGFILE" 2>&1
