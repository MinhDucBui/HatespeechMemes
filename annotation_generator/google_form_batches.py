import argparse
import os
import random
import shutil
import pandas as pd


BATCH_SIZE = 16
NUM_BATCHES = 25
PAGE_SIZE = 1
LANGUAGE = ["es"]
# LANGUAGE_FILTER = ["en", "de", "zh", "hi", "es"]
LANGUAGE_FILTER = ["es"]
random.seed(42)


def pop_elements_by_range(my_list, start_index, end_index):
    """
    Pops elements from my_list within the specified start and end indices.

    Args:
    my_list (list): The list from which to pop elements.
    start_index (int): The start index of the range (inclusive).
    end_index (int): The end index of the range (inclusive).

    Returns:
    list: A list of popped elements.
    """
    # Ensure the indices are within the valid range
    if start_index < 0 or end_index >= len(my_list) or start_index > end_index:
        raise ValueError("Invalid start or end index.")

    # Collect the indices to pop
    indices_to_pop = list(range(start_index, end_index))

    # Get and pop the elements
    popped_elements = [my_list.pop(index - i)
                       for i, index in enumerate(indices_to_pop)]

    return popped_elements


def redistribute_elements(groups, num_batches):
    # Initialize batches as empty lists
    batches = [[] for _ in range(num_batches)]
    current_batch = 0

    # Distribute elements in a round-robin fashion
    for group in groups:
        for element in group:
            batches[current_batch].append(element)
            current_batch = (current_batch + 1) % num_batches

    return batches


def restructure_list_of_lists(lists):
    # Flatten the list of lists into a single list
    flat_list = [item for sublist in lists for item in sublist]

    # Split the flattened list into chunks of 36 elements
    restructured_list = [flat_list[i:i + BATCH_SIZE]
                         for i in range(0, len(flat_list), BATCH_SIZE)]

    return restructured_list


def filter_batch(all_groups, filter_folder):
    df_filtered = []
    for language in LANGUAGE_FILTER:
        filter_language = os.path.join(
            filter_folder + language + ".csv")
        if os.path.isfile(filter_language):
            df_filtered.append(pd.read_csv(filter_language))
    df_filtered = pd.concat(df_filtered, ignore_index=True)
    image_ids = list(df_filtered["Image ID"].astype(str))

    new_all_groups = []
    for index_batch, group in enumerate(all_groups):
        new_all_groups.append([])
        for instance in group:
            instance_check = instance.split("/")[-1].split(".jpg")[0]
            if instance_check in image_ids:
                new_all_groups[index_batch].append(instance)
    new_all_groups = restructure_list_of_lists(new_all_groups)

    return new_all_groups


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--hatespeech', '-s', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/hatespeech_nonhate/images')
    parser.add_argument('--output', '-o', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/google_form_16_idk_new')
    parser.add_argument('--filter', '-f', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/prolific_annotations/filter_idk_labels_new/PRELIMINARY ')

    args = parser.parse_args()

    hatespeech_folder = args.hatespeech
    outpot_folder = args.output
    filter_folder = args.filter

    if LANGUAGE_FILTER != []:
        outpot_folder = outpot_folder + "_filtered"

    sourcing_folder = os.path.join(hatespeech_folder, "en")

    # 8 per group is going to be in
    all_groups = []
    # Loop over all files in the directory
    for group in os.listdir(sourcing_folder):
        group_path = os.path.join(sourcing_folder, group)
        instances = []
        if os.path.isdir(group_path):
            for instance_id in os.listdir(group_path):
                instances.append(os.path.join(group_path, instance_id))
            random.shuffle(instances)
            all_groups.append(instances)

    # Distribute the instances
    all_groups = redistribute_elements(all_groups, NUM_BATCHES)

    if filter_folder != "" and LANGUAGE_FILTER != []:
        all_groups = filter_batch(all_groups, filter_folder)

    for i, batch in enumerate(all_groups):
        print("Length of batch {}: {}".format(i, len(batch)))

    max_page = 4
    for language in LANGUAGE:
        if LANGUAGE_FILTER != []:
            outpot_folder_language = os.path.join(outpot_folder, LANGUAGE_FILTER[0] + "_main")
        else:
            outpot_folder_language = os.path.join(outpot_folder)
        os.makedirs(outpot_folder_language, exist_ok=True)
        page_index = 0
        for index_batch, group in enumerate(all_groups):
            for instance_index, instance in enumerate(group):

                instance = instance.replace("/en/", "/{}/".format(language))
                batch_folder = os.path.join(
                    outpot_folder_language, language, "batch_" + str(index_batch), str(page_index))
                os.makedirs(batch_folder, exist_ok=True)
                if instance is not None:
                    filename = instance.split("/")[-1]
                    # Copy each file from source to target
                    target_file = os.path.join(batch_folder, filename)
                    # copy2 preserves metadata
                    shutil.copy2(instance, target_file)
                    page_index = (page_index + 1) % max_page
