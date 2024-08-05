import os
import shutil


def restructure_folders(base_path):
    # List all directories in the base path
    all_batches = sorted([d for d in os.listdir(base_path) if os.path.isdir(
        os.path.join(base_path, d)) and d.startswith('batch_')])

    upload_count = 0
    for i in range(0, len(all_batches), 2):
        # Create a new upload folder
        upload_folder_name = f"upload_{upload_count}"
        upload_folder_path = os.path.join(base_path, upload_folder_name)
        os.makedirs(upload_folder_path, exist_ok=True)

        # Move the next two batch folders to the new upload folder
        for j in range(2):
            if i + j < len(all_batches):
                batch_folder_name = all_batches[i + j]
                batch_folder_path = os.path.join(base_path, batch_folder_name)
                shutil.move(batch_folder_path, os.path.join(
                    upload_folder_path, batch_folder_name))

        upload_count += 1


# Example usage:
# Replace with the actual path to the directory containing the batch folders
base_path = '/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/google_form/de'
restructure_folders(base_path)
