#!/bin/bash
# Submit a q2_campaign.sbatch wave SPREAD across named nodes, one pinned job each.
#
# Why pinning is necessary: this cluster rejects --cpus-per-task ("Please do not
# specify the number of CPUs. There is actually no limit."), so SLURM's CPU
# accounting does not see what a job actually runs. Left to itself it packed
# five 48-proc jobs onto one 96-core node (rlab2, 2026-08-19) while four whole
# nodes idled -- 240 processes timesharing 96 cores. Placement is therefore ours
# to make, via --nodelist, and the process count is ours to size, via xargs -P.
#
# Usage:
#   q2_submit_spread.sh SPLIT LEVEL MODEL AMBIENT TRIALS NSHARDS TLO PROCS PLAN...
# where each PLAN entry is node:jobcount, e.g. rlab7:11 rlab3:4
#
# Sizing rule: PROCS should be >= shards-per-job (NSHARDS / total jobs) so every
# job finishes its slice in a single wave, and jobcount*PROCS must stay under the
# node's core count with headroom left for other users.
#
# Example (560 eval shards over 28 jobs x 20 procs = one shard per core):
#   q2_submit_spread.sh eval 0.05 sine+ambient 0.09 20 560 0 20 \
#       rlab7:11 rlab3:4 ilab4:4 rlab4:4 ilab3:3 rlab2:2
set -euo pipefail

REPO=/common/home/dm1487/robotics_research/tripods/safe-control-gym
SPLIT=$1; LEVEL=$2; MODEL=$3; AMB=$4; TRIALS=$5; NSH=$6; TLO=$7; PROCS=$8
shift 8
[ $# -gt 0 ] || { echo "no node plan given" >&2; exit 2; }

# Memory: one collector process holds ~720 MB resident (measured on ilab2,
# 2026-08-19), so give each job its procs' worth plus roughly 60% headroom.
MEM_GB=$(( (PROCS * 720 * 16 / 10 + 999) / 1000 ))

NJOBS=0
for entry in "$@"; do NJOBS=$(( NJOBS + ${entry#*:} )); done
echo "plan: $NJOBS jobs x $PROCS procs = $(( NJOBS * PROCS )) procs, ${MEM_GB}G each"
echo "      $NSH $SPLIT shards -> ~$(( (NSH + NJOBS - 1) / NJOBS )) shards/job"

idx=0
for entry in "$@"; do
  node=${entry%%:*}; count=${entry#*:}
  for _ in $(seq 1 "$count"); do
    if [ "$idx" -ge "$NJOBS" ]; then break; fi
    sbatch --nodelist="$node" --mem="${MEM_GB}G" --chdir="$REPO" \
           "$REPO/q2_campaign.sbatch" \
           "$SPLIT" "$LEVEL" "$MODEL" "$AMB" "$TRIALS" "$NSH" \
           "$idx" "$NJOBS" "$PROCS" "$TLO" \
      | sed "s/\$/  -> $node (jobidx $idx)/"
    idx=$(( idx + 1 ))
  done
done
echo "submitted $idx jobs"
