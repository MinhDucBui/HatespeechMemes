import argparse
import shutil
import os

LANGUAGES = ["de", "en"]


def count_files_in_directory(directory):
    # List all files and directories in the specified directory
    files_and_dirs = os.listdir(directory)
    # Filter out directories, keep only files
    files = [f for f in files_and_dirs if os.path.isfile(
        os.path.join(directory, f))]
    return len(files)

def find_smallest_index(lst):
    # Check if the list is not empty
    if not lst:
        return None  # or raise an exception if preferred
    # Find the minimum element in the list
    min_value = min(lst)
    # Get the index of the minimum element
    min_index = lst.index(min_value)
    return min_index


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--image_folder', '-i', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/examples/output/user_')
    parser.add_argument('--output_path', '-o', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/google_form_nonhate_filter/')
    args = parser.parse_args()

    for language in LANGUAGES:
        image_folder = args.image_folder
        output_path = args.output_path
        output_path = output_path + language
        image_folder = image_folder + language
        test = 0
        # Iterate through all files in the directory
        for filename in os.listdir(output_path):
            # Construct the full file path
            batch_path = os.path.join(output_path, filename)
            if os.path.isdir(batch_path):
                length_batch = []
                batch_indices = []
                for batch_index in os.listdir(batch_path):
                    path_to_single_page = os.path.join(batch_path, batch_index)
                    if os.path.isdir(path_to_single_page):
                        batch_indices.append(int(batch_index))
                        length_batch.append(count_files_in_directory(path_to_single_page))

                # Pair the elements with their corresponding indices
                paired_list = list(zip(batch_indices, length_batch))
                sorted_pairs = sorted(paired_list)
                length_batch = [element for _, element in sorted_pairs]

                fill_index = find_smallest_index(length_batch)
                batch_indices.sort()

                for filename in os.listdir(image_folder):
                    instance_path = os.path.join(image_folder, filename)

                    if "1436.jpg" in instance_path:  # Asian
                        folder_index = 1
                    elif "1134290.jpg" in instance_path:  # French
                        folder_index = fill_index
                    elif "332838.jpg" in instance_path:  # Muslim Nonhate
                        folder_index = 3
                    elif "6167601.jpg" in instance_path:  # Attention Checker
                        folder_index = 2
                    elif "2061647.jpg" in instance_path:  # Native American
                        folder_index = 0
                    else:
                        continue

                    if folder_index >= len(batch_indices):
                        continue
                    filename_output = os.path.join(batch_path, str(
                        batch_indices[folder_index]), filename)
                    shutil.copy2(instance_path, filename_output)
