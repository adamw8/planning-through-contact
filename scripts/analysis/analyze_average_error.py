import argparse
import pickle

import numpy as np


def compute_average_errors(summary_path):
    """
    Compute and display the average successful translation and rotation errors
    from the given summary.pkl file.

    Args:
        summary_path (str): Path to the summary.pkl file.
    """
    # Load the summary.pkl file
    with open(summary_path, "rb") as f:
        summary = pickle.load(f)

    # Check if there are successful trials
    if len(summary["successful_trials"]) == 0:
        average_successful_trans_error = "N/A"
        average_successful_rot_error = "N/A"
    else:
        successful_translation_errors = []
        successful_rotation_errors = []

        # Compute errors for each successful trial
        for trial_idx in summary["successful_trials"]:
            successful_translation_errors.append(
                np.linalg.norm(summary["final_error"][trial_idx]["slider_error"][:2])
            )
            successful_rotation_errors.append(
                np.abs(summary["final_error"][trial_idx]["slider_error"][2])
            )

        # Compute averages
        average_successful_trans_error = np.mean(successful_translation_errors)
        average_successful_rot_error = np.mean(successful_rotation_errors)

    # Display results
    print("\n" + "=" * 50)
    print("=== Average Successful Errors ===")
    print(f"Number of Successful Trials: {len(summary['successful_trials'])}")
    print(
        f"Success Rate: {len(summary['successful_trials']) / len(summary['trial_times'])}"
    )
    print(
        f"Average Successful Translation Error: {100*average_successful_trans_error:.2f}cm"
    )
    print(
        f"Average Successful Rotation Error: {np.rad2deg(average_successful_rot_error):.2f}°"
    )
    print("=" * 50 + "\n")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Compute average successful errors from a summary.pkl file."
    )
    parser.add_argument("summary_path", type=str, help="Path to the summary.pkl file.")
    args = parser.parse_args()

    # Compute and display average errors
    compute_average_errors(args.summary_path)


if __name__ == "__main__":
    main()
