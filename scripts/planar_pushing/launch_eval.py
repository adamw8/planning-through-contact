import argparse
import copy
import csv
import os
import pickle
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Common arguments
CONFIG_DIR = "config/sim_config/sim_sim"
CONFIG_NAME = "gamepad_teleop.yaml"
BASE_COMMAND = [
    "python",
    "scripts/planar_pushing/run_sim_sim_eval.py",
    f"--config-dir={CONFIG_DIR}",
]
SUCCESS_RATES = {}

# ---------------------------------------------------------
# Example Usage:
# python launch_simulations.py --csv-path /path/to/jobs.csv --max-concurrent-jobs 8
#
# CSV file format:
# checkpoint_path,run_dir,config_name (optional)
# /path/to/checkpoint1.ckpt, data/test1, custom_config.yaml
# /path/to/checkpoint2.ckpt, data/test2
# ---------------------------------------------------------


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch multiple Hydra simulation commands concurrently."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to the CSV file containing checkpoint paths, run directories, and optional config names.",
    )
    parser.add_argument(
        "--max-concurrent-jobs",
        type=int,
        default=8,
        help="Maximum number of concurrent jobs (default: 8).",
    )
    return parser.parse_args()


def load_jobs_from_csv(csv_file):
    """Load checkpoint_path, run_dir, and optional config_name from a CSV file."""
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file '{csv_file}' does not exist.")

    jobs = []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            checkpoint_path = row.get("checkpoint_path", "").strip()
            run_dir = row.get("run_dir", "").strip()
            config_name = CONFIG_NAME
            optional_config_name = row.get("config_name", "")
            if optional_config_name is not None:
                optional_config_name = optional_config_name.strip()
                if optional_config_name != "":
                    config_name = optional_config_name

            # Eval single checkpoint
            if checkpoint_path.endswith(".ckpt"):
                assert os.path.exists(
                    checkpoint_path
                ), f"Checkpoint file '{checkpoint_path}' does not exist."
                checkpoint_file = os.path.basename(checkpoint_path)
                if checkpoint_path and run_dir:
                    jobs.append(
                        (checkpoint_path, f"{run_dir}/{checkpoint_file}", config_name)
                    )
            # Eval all checkpoints from the training run
            else:
                checkpoints_dir = os.path.join(checkpoint_path, "checkpoints")
                for checkpoint_file in os.listdir(checkpoints_dir):
                    if checkpoint_file.endswith(".ckpt"):
                        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_file)
                        jobs.append(
                            (
                                checkpoint_path,
                                os.path.join(run_dir, checkpoint_file),
                                config_name,
                            )
                        )
    return jobs


def get_best_success_rates(success_rates):
    """
    Extract the best success rate for each experiment.

    Args:
        success_rates (dict): Dictionary with keys as checkpoint paths and values as success rates.

    Returns:
        dict: Dictionary with the best success rate for each experiment.
    """
    grouped_experiments = {}
    for key, value in success_rates.items():
        experiment = os.path.dirname(key)
        if experiment not in grouped_experiments:
            grouped_experiments[experiment] = []
        grouped_experiments[experiment].append((key, value))

    best_sr = {}
    for experiment, checkpoints in grouped_experiments.items():
        best_checkpoint, best_rate = max(checkpoints, key=lambda x: x[1])
        best_sr[best_checkpoint] = best_rate

    return best_sr


def run_simulation(checkpoint_path, run_dir, config_name, remaining_jobs=None):
    """Run a single simulation with specified checkpoint, run directory, and config name."""
    command = BASE_COMMAND + [
        f"--config-name={config_name}",
        f'diffusion_policy_config.checkpoint="{checkpoint_path}"',
        f'hydra.run.dir="{run_dir}"',
    ]
    command_str = " ".join(command)

    print("Remaining jobs: ", remaining_jobs)
    print("\n" + "=" * 50)
    print(f"=== JOB START: {run_dir} ===")
    print(command_str)
    print("=" * 50 + "\n")

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"\n✅ Completed: {run_dir}")

        # Compute success rate
        summary_file = os.path.join(run_dir, "summary.pkl")
        with open(summary_file, "rb") as f:
            summary = pickle.load(f)
        success_rate = len(summary["successful_trials"]) / len(summary["trial_times"])
    else:
        print(f"\n❌ Failed: {run_dir}\nError: {result.stderr}")
        success_rate = None

    global SUCCESS_RATES
    SUCCESS_RATES[run_dir] = success_rate
    print("\n" + "=" * 50)
    print(f"=== JOB END: {run_dir} ===")
    print(f"Success Rate: {success_rate}")
    print("=" * 50 + "\n")


def main():
    args = parse_arguments()
    csv_file = args.csv_path
    max_concurrent_jobs = args.max_concurrent_jobs

    jobs = load_jobs_from_csv(csv_file)

    if not jobs:
        print("No valid jobs found in the CSV file. Please check the file.")
        return

    print(f"Loaded {len(jobs)} jobs from {csv_file}.")
    print(f"Running with up to {max_concurrent_jobs} concurrent jobs.\n")

    for job in jobs:
        output_dir = job[1]
        if os.path.exists(output_dir):
            print(
                f"Output directory '{output_dir}' already exists. Running this job will delete the existing contents."
            )
            resp = input("Run job anyways? [y/n]: ")
            if resp.lower() == "y":
                print("Deleting output directory...\n")
                shutil.rmtree(output_dir)
            else:
                print("Exiting...")
                return

    with ThreadPoolExecutor(max_workers=max_concurrent_jobs) as executor:
        futures = {}
        # print number of remaining jobs
        for checkpoint, run_dir, config_name in jobs:
            remaining_jobs = len(jobs) - len(futures)
            future = executor.submit(
                run_simulation, checkpoint, run_dir, config_name, remaining_jobs
            )
            futures[future] = (checkpoint, run_dir)
            time.sleep(2)  # try to avoid syncing issues (arbitrary_shape.sdf error)

        for future in as_completed(futures):
            checkpoint, run_dir = futures[future]
            try:
                future.result()
            except Exception as e:
                print(
                    f"❌ Error in job with checkpoint: {checkpoint}, run_dir: {run_dir}\n{e}"
                )

    # Print success rates sorted by keys
    print("\n" + "=" * 50)
    print("=== SUCCESS RATES ===")
    for run_dir in sorted(SUCCESS_RATES.keys()):
        success_rate = SUCCESS_RATES[run_dir]
        print(f"{run_dir}: {success_rate}")
    print("=" * 50 + "\n")

    # Print best success rate for each experiment
    best_sr = get_best_success_rates(copy.deepcopy(SUCCESS_RATES))
    print("\n" + "=" * 50)
    print("=== BEST SUCCESS RATES ===")
    for run_dir in sorted(best_sr.keys()):
        success_rate = best_sr[run_dir]
        print(f"{run_dir}: {success_rate}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
