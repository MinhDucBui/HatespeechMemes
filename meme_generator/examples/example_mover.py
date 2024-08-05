import argparse
import shutil
import os

LANGUAGES = ["en", "de"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--image_folder', '-i', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/examples/output/implicit_')
    parser.add_argument('--output_path', '-o', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/google_form_filtered/')
    args = parser.parse_args()

    for language in LANGUAGES:
        image_folder = args.image_folder
        output_path = args.output_path
        output_path = output_path + language
        image_folder = image_folder + language

        # Iterate through all files in the directory
        for filename in os.listdir(output_path):
            # Construct the full file path
            batch_path = os.path.join(output_path, filename)
            if os.path.isdir(batch_path):
                batch_indices = [int(batch_index) for batch_index in os.listdir(batch_path)
                                if os.path.isdir(os.path.join(batch_path, batch_index))]
                batch_indices.sort()

                for filename in os.listdir(image_folder):
                    instance_path = os.path.join(image_folder, filename)

                    if "1436.jpg" in instance_path:
                        folder_index = 3
                    elif "3208.jpg" in instance_path:
                        folder_index = 1
                    elif "332838.jpg" in instance_path:
                        folder_index = 0
                    elif "6453957.jpg" in instance_path:
                        folder_index = 5

                    # Attention Checker
                    elif "2061647.jpg" in instance_path:
                        folder_index = 2
                    elif "6167601.jpg" in instance_path:
                        folder_index = 4
                    else:
                        continue

                    if folder_index >= len(batch_indices):
                        continue
                    filename_output = os.path.join(batch_path, str(
                        batch_indices[folder_index]), filename)
                    shutil.copy2(instance_path, filename_output)
