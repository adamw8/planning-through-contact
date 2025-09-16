#!/bin/bash

# Usage
# LLsub ./submit_training.sh -s 20 -g volta:1

# Initialize and Load Modules
echo "[submit_training.sh] Loading modules and virtual environment"
source /etc/profile
module load anaconda/Python-ML-2025a

# Assume current directory is gcs-diffusion
# source .robodiff/bin/activate || echo "Training with anaconda/2023b module instead of venv"

# Set wandb to offline since Supercloud has no internet access
echo "[submit_training.sh] Setting wandb to offline"
wandb offline

# Export date, time, environment variables
DATE=`date +"%Y.%m.%d"`
TIME=`date +"%H.%M.%S"`
HYDRA_FULL_ERROR=1

echo "[submit_training.sh] Running training code..."
echo "[submit_training.sh] Date: $DATE"
echo "[submit_training.sh] Time: $TIME"

# CONFIG_DIR=config/planar_pushing/context_length_exp_adam_data_resnet_plus_transformer/hierarchy_basic_causal/
# CONFIG_NAME=16_obs.yaml
# HYDRA_RUN_DIR=data/outputs/context_length_exp_adam_data_resnet_plus_transformer/hierarchy_basic_causal/16_obs
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=/home/gridsan/aagarwal2/RLG/gcs-diffusion:$PYTHONPATH
export MOSEKLM_LICENSE_FILE=/home/gridsan/aagarwal2/mosek.lic
CSV_PATH=config/launch_eval_supercloud_2.txt

# CONFIG_DIR=config/planar_pushing/context_length_exp_adam_data_constant_model_size_init_encoder/2_encoder_freeze_then_resume/
# CONFIG_NAME=1_obs.yaml
# HYDRA_RUN_DIR=data/outputs/context_length_exp_adam_data_constant_model_size_init_encoder/2_encoder_freeze_then_resume/1_obs

# CONFIG_DIR=config/grasp_two_bins/resnet_plus_transformer/cls_token_only/
# CONFIG_NAME=5_obs.yaml
# HYDRA_RUN_DIR=data/outputs/grasp_two_bins/resnet_plus_transformer/cls_token_only/5_obs

python scripts/planar_pushing/launch_eval.py --csv-path $CSV_PATH --max-concurrent-jobs 10
