import os
import time
from datetime import datetime


def run_data_generation_script(
    config_dir, config_name, plans_dir, suppress_output=False
):
    seed = datetime.now().timestamp()
    command = (
        f"python scripts/planar_pushing/run_data_generation.py "
        f"--config-dir={config_dir} "
        f"--config-name {config_name} "
        f"data_collection_config.plans_dir={plans_dir} "
        f"data_collection_config.plan_config.seed={int(seed) % 1000} "
        f"multi_run_config.seed={int(seed) % 1000} "
    )

    if suppress_output:
        command += " > /dev/null 2>&1"

    os.system(command)


# Adjust these values
# ----------------------------------------------------------

# indice range is INCLUSIVE
start_index = 0
end_index = 59

config_dir = "config/sim_config"
config_name = "real_iiwa_sim_config.yaml"
plans_root = "/home/adam/workspace/planning-through-contact/trajectories/sim_tee_data"

suppress_output = False

# ----------------------------------------------------------

if __name__ == "__main__":
    # Loop through the indices and execute the command
    for i in range(start_index, end_index + 1):
        plans_dir = f"{plans_root}/run_{i}"
        print(plans_dir)
        start = time.time()
        run_data_generation_script(config_dir, config_name, plans_dir, suppress_output)
        print(
            f"Finished rendering plans for run {i} in {time.time() - start:.2f} seconds.\n"
        )
