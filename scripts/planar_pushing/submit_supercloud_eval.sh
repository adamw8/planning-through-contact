#!/bin/bash

# Usage
# LLsub sripts/planar_pushing/submit_supercloud_eval.sh -s 20 -g volta:1 # TODO: potentially just use cpu

# Initialize and Load Modules
echo "[submit_supercloud_eval.sh] Loading modules and virtual environment"
source /etc/profile
module load anaconda/2023a
export PYTHONPATH=~/workspace/ambient-diffusion-policy:$PYTHONPATH
echo $PYTHONPATH

# Activate virtual environment
echo "[submit_supercloud_eval.sh] Activating virtual environment"
cd /home/gridsan/awei/workspace/planning-through-contact
source activate /home/gridsan/awei/.conda/envs/planning_through_contact

# Set wandb to offline since Supercloud has no internet access
echo "[submit_supercloud_eval.sh] Setting wandb to offline"
wandb offline

# Export date, time, environment variables
DATE=`date +"%Y.%m.%d"`
TIME=`date +"%H.%M.%S"`
HYDRA_FULL_ERROR=1

echo "[submit_supercloud_eval.sh] Running eval code..."
echo "[submit_supercloud_eval.sh] Date: $DATE"
echo "[submit_supercloud_eval.sh] Time: $TIME"

# Record start time
START_TIME=$(date +%s)

# Create temporary config file
echo "[submit_supercloud_eval.sh] Creating temporary config file"

CONFIG_FILE="config/launch_eval_tmp.txt"
cat <<EOF > "$CONFIG_FILE"
checkpoint_path,run_dir,config_name
$HOME/workspace/ambient-diffusion-policy/data/outputs/ambient_diffusion/planar_pushing/ambient_loss/50_2000_t_min_05_sample/checkpoints/latest.ckpt, eval/ambient_diffusion/planar_pushing/test_sc, gamepad_teleop_carbon.yaml
EOF

echo "[submit_supercloud_eval.sh] Running eval command..."
python scripts/planar_pushing/launch_eval.py --csv-path "$CONFIG_FILE" --max-concurrent-jobs 8 --num-trials 50 50 100

# Remove temporary config file
rm "$CONFIG_FILE"

# Calculate and print duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))
SECONDS=$((DURATION % 60))

echo "[submit_supercloud_eval.sh] Job completed in ${HOURS}h ${MINUTES}m ${SECONDS}s"
