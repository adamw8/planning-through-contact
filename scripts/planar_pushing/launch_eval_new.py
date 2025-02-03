import argparse
import copy
import csv
import os
import pickle
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from queue import Queue

from scipy.integrate import quad
from scipy.stats import beta

# Common arguments
CONFIG_DIR = "config/sim_config/sim_sim"
CONFIG_NAME = "gamepad_teleop.yaml"
BASE_COMMAND = [
    "python",
    "scripts/planar_pushing/run_sim_sim_eval.py",
    f"--config-dir={CONFIG_DIR}",
]

# ---------------------------------------------------------
# Example Usage:
# python launch_simulations.py --csv-path /path/to/jobs.csv --max-concurrent-jobs 8
#
# CSV file format:
# checkpoint_path,run_dir,config_name (optional)
# /path/to/checkpoint1.ckpt, data/test1, custom_config.yaml
# /path/to/checkpoint2.ckpt, data/test2
# ---------------------------------------------------------


@dataclass
class JobConfig:
    checkpoint_path: str
    run_dir: str
    config_name: str
    num_trials: int = -1
    seed: int = 0
    continue_flag: bool = False

    def __str__(self):
        return f"checkpoint_path={self.checkpoint_path}, run_dir={self.run_dir}, config_name={self.config_name}, num_trials={self.num_trails}, seed={self.seed}, continue_flag={self.continue_flag}"

    def __repr__(self):
        return str(self)


@dataclass
class JobResult:
    num_successful_trials: int
    num_trials: int

    def __post_init__(self):
        self.success_rate = self.num_successful_trials / self.num_trials

    def __str__(self):
        return f"num_successful_trials={self.num_successful_trials}, num_trials={self.num_trials}, success_rate={self.success_rate}"

    def __repr__(self):
        return str(self)


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
    """Load checkpoint groups, where each group consists of one or more checkpoints."""
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file '{csv_file}' does not exist.")

    job_groups = []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            checkpoint_path = row.get("checkpoint_path", "").strip()
            run_dir = row.get("run_dir", "").strip()
            config_name = row.get("config_name", CONFIG_NAME).strip()

            # If evaluating a single checkpoint, create a single-element group
            if checkpoint_path.endswith(".ckpt"):
                assert os.path.exists(
                    checkpoint_path
                ), f"Checkpoint file '{checkpoint_path}' does not exist."
                checkpoint_file = os.path.basename(checkpoint_path)

                job_config = JobConfig(
                    checkpoint_path=checkpoint_path,
                    run_dir=f"{run_dir}/{checkpoint_file}",
                    config_name=config_name,
                    seed=0,
                    continue_flag=False,
                )
                job_groups.append([job_config])

            # If evaluating all checkpoints from a training run, create a group
            else:
                checkpoint_group = []
                checkpoints_dir = os.path.join(checkpoint_path, "checkpoints")
                for checkpoint_file in os.listdir(checkpoints_dir):
                    if checkpoint_file.endswith(".ckpt"):
                        full_checkpoint_path = os.path.join(
                            checkpoints_dir, checkpoint_file
                        )
                        job_config = JobConfig(
                            checkpoint_path=full_checkpoint_path,
                            run_dir=os.path.join(run_dir, checkpoint_file),
                            config_name=config_name,
                            seed=0,
                            continue_flag=False,
                        )
                        checkpoint_group.append(job_config)
                job_groups.append(checkpoint_group)

    return job_groups


def run_simulation(job_config):
    """Run a single simulation with specified checkpoint, run directory, and config name."""
    checkpoint_path = job_config.checkpoint_path
    run_dir = job_config.run_dir
    config_name = job_config.config_name
    num_trials = job_config.num_trials
    seed = job_config.seed
    continue_flag = job_config.continue_flag
    assert num_trials > 0, "num_trials must be greater than 0"

    # TODO: provide overrides here
    command = BASE_COMMAND + [
        f"--config-name={config_name}",
        f'diffusion_policy_config.checkpoint="{checkpoint_path}"',
        f'hydra.run.dir="{run_dir}"',
        f"multi_run_config.seed={seed}",
        f"multi_run_config.num_runs={num_trials}",
        f"++continue_eval={continue_flag}",
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

    print("\n" + "=" * 50)
    print(f"=== JOB END: {run_dir} ===")
    print(f"Success Rate: {success_rate}")
    print("=" * 50 + "\n")

    if success_rate is None:
        return None
    else:
        return JobResult(
            num_successful_trials=len(summary["successful_trials"]),
            num_trials=len(summary["trial_times"]),
        )


def get_eval_command(job_config):
    """Get the command to evaluate a single checkpoint."""
    checkpoint_path = job_config.checkpoint_path
    run_dir = job_config.run_dir
    config_name = job_config.config_name
    seed = job_config.seed
    continue_flag = job_config.continue_flag

    command = BASE_COMMAND + [
        f"--config-name={config_name}",
        f'diffusion_policy_config.checkpoint="{checkpoint_path}"',
        f"multi_run_config.seed={seed}",
        f'hydra.run.dir="{run_dir}"',
    ]
    if continue_flag:
        command.append("++continue_eval=true")
    command_str = " ".join(command)
    return command_str


def validate_job_groups(job_groups):
    if not job_groups:
        print("No valid jobs found in the CSV file. Please check the file.")
        return False

    # Sure there are no duplicate logging directories in the jobs list
    logging_dirs = []
    for group in job_groups:
        for job in group:
            logging_dirs.append(job.run_dir)
    if len(logging_dirs) != len(set(logging_dirs)):
        print("Duplicate logging directories found in the jobs list.")
        return False

    # Double check if output directories already exist
    for group in job_groups:
        for job in group:
            output_dir = job.run_dir
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
                    return False

    return True


def print_diagnostic_info(job_groups, max_concurrent_jobs, num_trials, drop_threshold):
    num_jobs = sum([len(group) for group in job_groups])

    print("\nDiagnostic Information:")
    print("=======================")
    print(
        f"Evaluating {len(job_groups)} training runs, consistenting of {num_jobs} checkpoints"
    )
    print(f"Running with {max_concurrent_jobs} jobs")
    print(f"Checkpoints will be compared at {num_trials} trials.")
    print(
        f"During each comparison, if the probability that a checkpoint is better than the current best checkpoint is less than {drop_threshold}, the checkpoint will be dropped."
    )
    print(f"The best checkpoints will be evaluated for {sum(num_trials)} trials.")
    print("\nTraining run details:")

    for group in job_groups:
        training_dir = os.path.dirname(
            group[0].checkpoint_path.split("/checkpoints")[0]
        )
        run_dir = os.path.dirname(group[0].run_dir)
        config_name = group[0].config_name
        print("------------------------------")
        print(f"Training Run: {training_dir}")
        print("Checkpoints:")
        for i, job in enumerate(group):
            print(f"  {i+1}. {os.path.basename(job.checkpoint_path)}")
        print(f"Eval directory: {job.run_dir}")
        print(f"Config Name: {job.config_name}")
        print()
    print()


def prob_p1_greater_p2(n1, N1, n2, N2):
    """
    Computes P(p1 > p2) where:
    - n1, N1: Successes and trials for p1
    - n2, N2: Successes and trials for p2

    Returns:
    - Probability that p1 > p2
    """

    # Numerical integration
    alpha1, beta1 = n1 + 1, N1 - n1 + 1
    alpha2, beta2 = n2 + 1, N2 - n2 + 1

    def cdf_p1(x):
        return beta.cdf(x, alpha1, beta1)

    def pdf_p2(x):
        return beta.pdf(x, alpha2, beta2)

    # p(p1 > p2) = int_0^1 cdf_p1(x) * pdf_p2(x) dx
    integral, _ = quad(lambda x: (1 - cdf_p1(x)) * pdf_p2(x), 0, 1)
    return integral


def main():
    args = parse_arguments()
    csv_file = args.csv_path
    max_concurrent_jobs = args.max_concurrent_jobs
    num_trials = [1, 1, 1]
    drop_threshold = 0.05

    job_groups = load_jobs_from_csv(csv_file)
    if not validate_job_groups(job_groups):
        return
    print_diagnostic_info(job_groups, max_concurrent_jobs, num_trials, drop_threshold)

    all_jobs = [job for group in job_groups for job in group]

    for i, trial in enumerate(num_trials):
        for j, job in enumerate(all_jobs):
            job.num_trials = trial
            if i != 0:
                job.continue_flag = True
        with ThreadPoolExecutor(max_workers=max_concurrent_jobs) as executor:
            futures = {}
            for job in all_jobs:
                future = executor.submit(run_simulation, job)
                futures[future] = job
                # time.sleep(1) # prevent syncing issues with arbitrary_shape.sdf

            for future in as_completed(futures):
                print(future)
                print(future.result())

    print("\n✅ All jobs finished.")


if __name__ == "__main__":
    main()
