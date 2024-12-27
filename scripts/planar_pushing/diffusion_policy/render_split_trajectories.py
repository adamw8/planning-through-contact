import os
from datetime import datetime

from tqdm import tqdm


def run_data_generation_script(config_name, config_dir, plans_dir):
    seed = datetime.now().timestamp()
    command = (
        f"python scripts/planar_pushing/diffusion_policy/run_data_generation.py "
        f"--config-dir {config_dir} "
        f"--config-name {config_name} "
        f"data_collection_config.plans_dir={plans_dir} "
        f"data_collection_config.plan_config.seed={int(seed) % 1000} "
        # f"multi_run_config.seed={int(seed) % 1000} "
    )

    os.system(command)


# Specify the range of indices
start_index = 2
end_index = 99  # Adjust this value as needed
config_dir = "config/sim_config/symmetries_project"
config_name = "baseline.yaml"
plans_root = f"trajectories/sim_box_data"

if __name__ == "__main__":
    # Loop through the indices and execute the command
    for i in range(start_index, end_index + 1):
        plans_dir = f"{plans_root}/run_{i}"
        print(plans_dir)
        run_data_generation_script(config_name, config_dir, plans_dir)
