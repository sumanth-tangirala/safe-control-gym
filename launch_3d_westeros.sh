#!/bin/bash
source /common/home/st1122/miniforge3/etc/profile.d/conda.sh
conda activate scg
cd /common/home/st1122/Projects/safe-control-gym
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
SEED=$1; TILT=$2
if [ "$TILT" = "full" ]; then FLAG=""; TAG="full"; else FLAG="--max_init_tilt_deg $TILT"; TAG="$TILT"; fi
nohup python3 train_quadrotor_3d_flip.py \
  --output_dir models/quad3d_s${SEED}_t${TAG} \
  --max_env_steps 1000000 --seed ${SEED} \
  --log_interval 5000 --save_interval 20000 --eval_interval 25000 \
  $FLAG > slurm_logs/flip3d/s${SEED}_t${TAG}.log 2>&1 &
echo "launched seed=$SEED tilt=$TAG pid=$!"
