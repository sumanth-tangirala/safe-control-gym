#!/bin/bash
#SBATCH --partition=unlimited
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --job-name=q2collect
#SBATCH --output=campaign_%x_%j.log
# Args: split level model ambient(or none) trials nshards jobidx njobs procs
# Runs shards jobidx, jobidx+njobs, ... concurrently, PROCS at a time.
# Idempotent: the collector skips existing --out files, so resubmit freely.
cd /common/home/dm1487/robotics_research/tripods/safe-control-gym
# Single-core shards: BLAS threading off, or 48 procs x 64 BLAS threads
# blows RLIMIT_NPROC (2000) and numpy dies at import.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_NUM_THREADS=1
# Temp dirs go to node-local NVMe, not the shared mount. pybullet unpacks its
# URDF assets into TMPDIR on every env reset; with 48 procs/node against an NFS
# dir holding 20k entries this returned EACCES and killed a whole wave
# (jobs 212074-212077, 2026-08-19).
export TMPDIR=/tmp/dm1487_q2_${SLURM_JOB_ID:-$$}
mkdir -p "$TMPDIR"
trap 'rm -rf -- "$TMPDIR"' EXIT
SPLIT=$1; LEVEL=$2; MODEL=$3; AMB=$4; TRIALS=$5; NSH=$6; JOB=$7; NJOBS=$8; PROCS=$9; TLO=${10:-0}
OUT=/common/users/dm1487/q2_corridor_shards
AMBFLAG=""; TAG="none"
if [ "$AMB" != "none" ]; then AMBFLAG="--ambient $AMB"; TAG=$AMB; fi
seq $JOB $NJOBS $((NSH-1)) | xargs -P $PROCS -I{} \
  /common/users/dm1487/envs/arcmg/bin/python3 q2_corridor_collect.py \
    --split $SPLIT --level $LEVEL --model $MODEL $AMBFLAG --trials $TRIALS \
    --trial_lo $TLO --shard {} --nshards $NSH \
    --out $OUT/${SPLIT}_L${LEVEL}_A${TAG}_k${TLO}-${TRIALS}_s{}.npz
