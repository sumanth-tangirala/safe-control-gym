#!/bin/bash
#SBATCH --partition=unlimited
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --job-name=q2topup
#SBATCH --output=topup_%x_%j.log
# Args: level ambient jobidx njobs procs
# Repairs the shards collected before the reachability shortcut was gated on
# the ambient term (2026-08-19). For each existing k0-20 shard it writes the
# complementary k1-20 window holding the flights the shortcut skipped; the
# reducer sums the two. Only shards that exist are touched, so this never
# invents a shard the collector has not produced.
# Idempotent: q2_corridor_topup.py skips an --out that already exists.
cd /common/home/dm1487/robotics_research/tripods/safe-control-gym
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_NUM_THREADS=1
# Node-local temp: pybullet unpacks URDFs into TMPDIR every env reset, and the
# shared mount returned EACCES under that load (jobs 212074-212077).
export TMPDIR=/tmp/dm1487_q2t_${SLURM_JOB_ID:-$$}
mkdir -p "$TMPDIR"
trap 'rm -rf -- "$TMPDIR"' EXIT
LEVEL=$1; AMB=$2; JOB=$3; NJOBS=$4; PROCS=$5
OUT=/common/users/dm1487/q2_corridor_shards
ls $OUT/eval_L${LEVEL}_A${AMB}_k0-20_s*.npz \
  | sed -E 's/.*_s([0-9]+)\.npz/\1/' | sort -n \
  | awk -v j="$JOB" -v n="$NJOBS" 'NR % n == j' \
  | xargs -P "$PROCS" -I{} \
      /common/users/dm1487/envs/arcmg/bin/python3 q2_corridor_topup.py \
        --in  $OUT/eval_L${LEVEL}_A${AMB}_k0-20_s{}.npz \
        --out $OUT/eval_L${LEVEL}_A${AMB}_k1-20_s{}.npz
