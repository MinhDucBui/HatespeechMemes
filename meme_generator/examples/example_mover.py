import argparse
import shutil
import os

LANGUAGES = ["zh"]
NAME_FOLDER = ["de_", "en_", "es_", "hi_", "zh_"]
MAX_LEN = 16


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
    min_indices = [i for i, value in enumerate(lst) if value == min_value]

    return min_indices


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--image_folder', '-i', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/examples/output/user_')
    parser.add_argument('--output_path', '-o', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/google_form_16_nodisagreement_filtered/all/all_zh/')
    args = parser.parse_args()

    # TODO: 
    for language in NAME_FOLDER:
        image_folder = args.image_folder
        output_path = args.output_path
        output_path = output_path + language + LANGUAGES[0]
        image_folder = image_folder + LANGUAGES[0]
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
                #sorted_pairs = sorted(paired_list)
                length_batch = [element for _, element in paired_list]
                sum_batch = sum(length_batch)

                min_indices = find_smallest_index(length_batch)
                #batch_indices.sort()
                #min_indices.sort()

                for filename in os.listdir(image_folder):
                    instance_path = os.path.join(image_folder, filename)

                    if "1436.jpg" in instance_path:  # Asian
                        folder_index = min_indices[2]
                    elif "1134290.jpg" in instance_path:  # French
                        if sum_batch == MAX_LEN:
                            folder_index = min_indices[3]
                        else:
                            continue
                    elif "332838.jpg" in instance_path:  # Muslim Nonhate
                        folder_index = min_indices[0]
                    elif "6167601.jpg" in instance_path:  # Attention Checker
                        folder_index = min_indices[1]
                    else:
                        continue

                    if folder_index >= len(batch_indices):
                        continue
                    filename_output = os.path.join(batch_path, str(
                        batch_indices[folder_index]), filename)
                    shutil.copy2(instance_path, filename_output)
