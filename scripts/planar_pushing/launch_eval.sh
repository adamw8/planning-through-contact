#!/bin/bash

# Usage
# LLsub ./scripts/planar_pushing/launch_eval.sh -s 20 -g volta:1

# Initialize and Load Modules
echo "[launch_eval.sh] Loading modules and virtual environment"
source /etc/profile
module load anaconda/2023a
source activate /home/gridsan/awei/.conda/envs/planning_through_contact
export PYTHONPATH=~/workspace/gcs-diffusion:${PYTHONPATH}

# Set wandb to offline since Supercloud has no internet access
echo "[launch_eval.sh] Setting wandb to offline"
wandb offline

# Export date, time, environment variables
DATE=`date +"%Y.%m.%d"`
TIME=`date +"%H.%M.%S"`
HYDRA_FULL_ERROR=1

echo "[launch_eval.sh] Running eval code..."
echo "[launch_eval.sh] Date: $DATE"
echo "[launch_eval.sh] Time: $TIME"

CONFIG_FILE="config/launch_eval_baseline.txt"
NUM_JOBS=4

# Eval command
python scripts/planar_pushing/launch_eval.py --csv-path $CONFIG_FILE --max-concurrent-jobs $NUM_JOBS
