import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np


def plot_success_rate(summary_file):
    """
    Plot the success rate vs the number of trials from a summary.pkl file.

    Parameters:
        summary_file (str): Path to the summary.pkl file.
    """
    # Load the summary.pkl file
    with open(summary_file, "rb") as f:
        summary = pickle.load(f)

    # Compute success rate for each trial
    max_trial = len(summary["trial_result"])
    success_rate = []
    total_success = 0

    for trial, result in enumerate(summary["trial_result"]):
        if result == "success":
            total_success += 1
        success_rate.append(total_success / (trial + 1))

    # Plot success rate vs number of trials
    trials = np.arange(1, max_trial + 1)
    final_success_rate = success_rate[-1] if success_rate else 0

    plt.figure(figsize=(10, 6))
    plt.plot(trials, success_rate)
    plt.axhline(
        y=final_success_rate,
        color="r",
        linestyle="--",
        label=f"Final Success Rate: {final_success_rate:.2f}",
    )
    plt.title("Success Rate vs. Number of Trials")
    plt.xlabel("Number of Trials")
    plt.ylabel("Success Rate")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    """
    Example usage:
    python scripts/plots/make_lln_plot.py eval/sim_sim/baseline/150/summary.pkl
    """
    parser = argparse.ArgumentParser(
        description="Plot Success Rate vs. Number of Trials from a summary.pkl file."
    )
    parser.add_argument("summary_file", type=str, help="Path to the summary.pkl file")

    args = parser.parse_args()
    plot_success_rate(args.summary_file)
