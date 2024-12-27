import os
import re
import shutil

# # Path to the directory containing subdirectories
# base_dir = "trajectories/sim_box_data_seed_1"

# # Number of subdirectories to move into each "run" directory
# batch_size = 50

# # Create the "run" directories if they don't exist
# for i in range(0, len(os.listdir(base_dir)), batch_size):
#     run_dir = os.path.join(base_dir, f"run_{i // batch_size}")
#     os.makedirs(run_dir, exist_ok=True)

# # Move subdirectories into the "run" directories
# i = 0
# for subdir in sorted(
#     [
#         name
#         for name in os.listdir(base_dir)
#         if not name.endswith("yaml") and not name.endswith("txt")
#     ],
#     key=lambda x: int(x.split("_")[1]),
# ):
#     if subdir.startswith("traj"):
#         source = os.path.join(base_dir, subdir)
#         destination = os.path.join(base_dir, f"run_{i // batch_size}")
#         shutil.move(source, destination)
#         i += 1

# print("All subdirectories moved into their respective 'run' directories.")

## Script for shifting run directories

# def rename_run_directories_with_offset(main_directory, offset):
#     """
#     Rename all subdirectories named 'run_{i}' to 'run_{i+offset}'.

#     Parameters:
#         main_directory (str): Path to the main directory.
#         offset (int): Offset to add to the numeric part of the directory name.
#     """
#     # List all entries in the main directory
#     entries = os.listdir(main_directory)

#     for entry in entries:
#         entry_path = os.path.join(main_directory, entry)

#         # Check if the entry is a directory and matches the pattern 'run_{i}'
#         if os.path.isdir(entry_path):
#             match = re.match(r"^run_(\d+)$", entry)
#             if match:
#                 old_index = int(match.group(1))
#                 new_index = old_index + offset
#                 new_name = f"run_{new_index}"
#                 new_path = os.path.join(main_directory, new_name)

#                 # Rename the directory
#                 os.rename(entry_path, new_path)
#                 print(f"Renamed: {entry} → {new_name}")

#     print("✅ All matching subdirectories have been renamed.")

# # Example usage
# if __name__ == "__main__":
#     main_directory = "trajectories/sim_box_data_seed_1"
#     offset = 50  # Change the offset value as needed

#     rename_run_directories_with_offset(main_directory, offset)
