import argparse
import os
import random
import shutil
import pandas as pd

BATCH_SIZE = 36
PAGE_SIZE = 6
LANGUAGE = ["en", "de"]
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


# Hardcoded: Because we have 45 groups, 8 instances. Distribute over 10 batches.
def distribute_instances(all_groups):
    # Number of batches
    num_batches = 10

    # Initialize batches
    batches = [[] for _ in range(num_batches)]

    # Distribute instances to batches
    for group in all_groups:
        for idx, instance in enumerate(group):
            # Determine which batch to put this instance in
            batch_number = (idx % num_batches)
            batches[batch_number].append(instance)

    for batch_number in range(8, 10):
        instance_range = 0
        if batch_number == 8:
            for group_index in range(4):
                elements = pop_elements_by_range(
                    batches[group_index], instance_range*9, (instance_range*9+9))
                batches[batch_number] += elements
                instance_range += 1
        elif batch_number == 9:
            for group_index in range(4, 8):
                elements = pop_elements_by_range(
                    batches[group_index], instance_range*9, (instance_range*9+9))
                batches[batch_number] += elements
                instance_range += 1
    all_a = []
    for i, batch in enumerate(batches):
        all_a += batch
        assert len(
            batch) == 36, f"Batch {i} has {len(batch)} instances instead of 36."
    return batches

def restructure_list_of_lists(lists):
    # Flatten the list of lists into a single list
    flat_list = [item for sublist in lists for item in sublist]
    
    # Split the flattened list into chunks of 36 elements
    restructured_list = [flat_list[i:i + 36] for i in range(0, len(flat_list), 36)]
    
    return restructured_list


def filter_batch(all_groups, filter_folder):
    if filter_folder != "":
        df_filtered = []
        for language in LANGUAGE:
            filter_language = os.path.join(
                filter_folder, "PRELIM_" + language + ".csv")
            df_filtered.append(pd.read_csv(filter_language))
        df_filtered = pd.concat(df_filtered, ignore_index=True)
        image_ids = list(df_filtered["Image ID"].astype(str))

    new_all_groups = []
    for index_batch, group in enumerate(all_groups):
        new_all_groups.append([])
        for instance in group:
            if filter_folder != "":
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
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/google_form_nonhate')
    parser.add_argument('--filter', '-f', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/annotation_evaluation/data/filter/data')

    args = parser.parse_args()

    hatespeech_folder = args.hatespeech
    outpot_folder = args.output
    filter_folder = args.filter

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
    all_groups = distribute_instances(all_groups)

    if filter_folder != "":
        all_groups = filter_batch(all_groups, filter_folder)

    for language in LANGUAGE:
        page_index = 0
        for index_batch, group in enumerate(all_groups):
            for instance_index, instance in enumerate(group):

                instance = instance.replace("/en/", "/{}/".format(language))
                batch_folder = os.path.join(
                    outpot_folder, language, "batch_" + str(index_batch), str(page_index))
                os.makedirs(batch_folder, exist_ok=True)
                if instance is not None:
                    filename = instance.split("/")[-1]
                    # Copy each file from source to target
                    target_file = os.path.join(batch_folder, filename)
                    # copy2 preserves metadata
                    shutil.copy2(instance, target_file)

                    if (instance_index+1) % 6 == 0:
                        page_index += 1
