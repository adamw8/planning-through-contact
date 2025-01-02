import argparse
import csv
import os
import pickle
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Common arguments
CONFIG_DIR = "config/sim_config"
CONFIG_NAME = "gamepad_teleop.yaml"
BASE_COMMAND = [
    "python",
    "scripts/planar_pushing/run_sim_sim_eval.py",
    f"--config-dir={CONFIG_DIR}",
    f"--config-name={CONFIG_NAME}",
]
SUCCESS_RATES = {}

# ---------------------------------------------------------
# Example Usage:
# python launch_simulations.py --csv-path /path/to/jobs.csv --max-concurrent-jobs 8
#
# CSV file format:
# checkpoint_path,run_dir
# /path/to/checkpoint1.ckpt, data/test1
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
        help="Path to the CSV file containing checkpoint paths and run directories.",
    )
    parser.add_argument(
        "--max-concurrent-jobs",
        type=int,
        default=8,
        help="Maximum number of concurrent jobs (default: 8).",
    )
    return parser.parse_args()


def load_jobs_from_csv(csv_file):
    """Load checkpoint_path and run_dir from a CSV file."""
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file '{csv_file}' does not exist.")

    jobs = []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            checkpoint_path = row.get("checkpoint_path", "").strip()
            run_dir = row.get("run_dir", "").strip()

            # Eval single checkpoint
            if checkpoint_path.endswith(".ckpt"):
                assert os.path.exists(
                    checkpoint_path
                ), f"Checkpoint file '{checkpoint_path}' does not exist."
                checkpoint_file = os.path.basename(checkpoint_path)
                if checkpoint_path and run_dir:
                    jobs.append((checkpoint_path, f"{run_dir}/{checkpoint_file}"))
            # Eval all checkpoints from the training run
            else:
                checkpoints_dir = os.path.join(checkpoint_path, "checkpoints")
                for checkpoint_file in os.listdir(checkpoints_dir):
                    if checkpoint_file.endswith(".ckpt"):
                        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_file)
                        jobs.append(
                            (checkpoint_path, os.path.join(run_dir, checkpoint_file))
                        )
    return jobs


def run_simulation(checkpoint_path, run_dir):
    """Run a single simulation with specified checkpoint and run directory."""
    command = BASE_COMMAND + [
        f'diffusion_policy_config.checkpoint="{checkpoint_path}"',
        f'hydra.run.dir="{run_dir}"',
    ]
    command_str = " ".join(command)

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
        for checkpoint, run_dir in jobs:
            future = executor.submit(run_simulation, checkpoint, run_dir)
            futures[future] = (checkpoint, run_dir)
            time.sleep(1) # try to avoid syncing issues (arbitrary_shape.sdf error)
        
        for future in as_completed(futures):
            checkpoint, run_dir = futures[future]
            try:
                future.result()
            except Exception as e:
                print(
                    f"❌ Error in job with checkpoint: {checkpoint}, run_dir: {run_dir}\n{e}"
                )

    # print success rates
    print("\n" + "=" * 50)
    print("=== SUCCESS RATES ===")
    for run_dir, success_rate in SUCCESS_RATES.items():
        print(f"{run_dir}: {success_rate}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
