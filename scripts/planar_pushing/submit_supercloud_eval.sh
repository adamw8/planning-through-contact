#!/bin/bash

# Usage (cpu nodes)
# LLsub scripts/planar_pushing/submit_supercloud_eval.sh -s 24

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
# TODO: potentially do this step outside of this script
echo "[submit_supercloud_eval.sh] Creating temporary config file"

CONFIG_FILE="config/launch_eval_tmp.txt"
CHECKPOINT_PATH="~/workspace/ambient-diffusion-policy/data/outputs/ambient_diffusion/planar_pushing/denoising_loss/sigma_min/50_2000_t_min_04_epsilon/checkpoints/epoch=0030-val_loss_0=0.0269-val_ddim_mse_0=0.0002.ckpt"
RUN_DIR="eval/ambient_diffusion/planar_pushing/test_sc"
CONFIG_NAME="gamepad_teleop_carbon.yaml"

rm "$CONFIG_FILE"
cat <<EOF > "$CONFIG_FILE"
checkpoint_path,run_dir,config_name
$CHECKPOINT_PATH, $RUN_DIR, $CONFIG_NAME
EOF

echo "[submit_supercloud_eval.sh] Running eval command..."
python scripts/planar_pushing/launch_eval.py --csv-path "$CONFIG_FILE" --device "cpu" --max-concurrent-jobs 1 --num-trials 50 50 100

# Remove temporary config file
rm "$CONFIG_FILE"

# Calculate and print duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))
SECONDS=$((DURATION % 60))

echo "[submit_supercloud_eval.sh] Job completed in ${HOURS}h ${MINUTES}m ${SECONDS}s"
