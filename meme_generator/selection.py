import argparse
import pandas as pd
import os
from tqdm import tqdm
from langdetect import detect, LangDetectException
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

LANGUAGES = ["de"]
SIZE = 100
CHOSEN_TEMPLATES = ["advice-hitler",
                    "african-children-dancing",
                    "american-pride-eagle",
                    "angry-black-woman",
                    "asinine-america",
                    "obama",
                    "obama laughing",
                    "Rich Men Laughing",
                    "scumbag-god",
                    "Successful Mexican"]

CHOSEN_TEMPLATES = ["american pride eagle",
                    "laughing girls",
                    "african children dancing",
                    "provincial man",
                    "obama laughing"]

CHOSEN_TEMPLATES = [template.replace("-", " ").lower()
                    for template in CHOSEN_TEMPLATES]


def detect_lang(input_str):
    try:
        return detect(input_str)
    except LangDetectException:
        return "Error"


def filter_english(df):
    df['caption_modified'] = df['caption'].str.replace('<sep>', '')
    df['caption_modified'] = df['caption_modified'].str.replace('<emp>', '')
    df['caption_modified'] = df['caption_modified'].str.replace('  ', ' ')
    df['caption_modified'] = df['caption_modified'].str.lower()
    tqdm.pandas()
    df['en_detect'] = df['caption_modified'].progress_apply(detect_lang)
    df = df[df['en_detect'] == "en"]
    return df


def collect_results(df):
    all_captions = []
    # Save Selection
    for _, row in df.iterrows():
        link = row["template"]
        instance_id = row["unique_id"]
        top = row["caption"].split("<sep>")[0].strip()
        bottom = row["caption"].split("<sep>")[1].strip()
        text_original = top + ' ' + "<sep>" + ' ' + bottom
        all_captions.append(
            f'{link}\t{instance_id}\t{text_original}\t{text_original}\n')

    return all_captions


def load_text_file(filename, headers):
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Process each line
    modified_lines = []
    for line in lines:
        # Split the line into columns using tab as the delimiter
        columns = line.split('\t')
        if len(columns) > 2:
            third_column = columns[2]
            # Count the number of double quotes in the third column
            quote_count = third_column.count('"')

            # Only remove quotes if the count is odd
            if quote_count % 2 != 0:
                third_column = third_column.replace(
                    '"', '')  # Remove all double quotes
                # Update the third column with the modified text
                columns[2] = third_column
        # Rejoin the columns into a single line
        modified_lines.append(columns)

    df = pd.DataFrame(modified_lines, columns=headers)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--memes', '-m', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/memes')
    parser.add_argument('--generate_folder', '-g', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/generated_memes')

    args = parser.parse_args()

    folder_path = args.memes
    output_folder = args.generate_folder

    if "existing" in output_folder:
        existing = args.generate_folder
    else:
        existing = None
        CHOSEN_TEMPLATES = None

    # Output Path
    output_path = os.path.join(output_folder, "caption_translation")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_language_file = os.path.join(output_path, "en.txt")

    # Replace with your actual column names
    headers = ['template', 'instance_id', 'caption']

    df_original = load_text_file(os.path.join(folder_path, "captions.txt"),
                                 headers=headers)

    # If existing, add this:
    if existing:
        df_original['unique_id'] = range(1, len(df_original) + 1)
    else:
        df_original['unique_id'] = df_original["instance_id"].astype(int)
    df_original = df_original.drop_duplicates()

    # ToDo: meme_generator
    # Clean the text file from odd number of '""

    df_original['template_normalized'] = df_original['template'].str.split(
        '_variant=').str[0]
    TEMPLATES = df_original['template_normalized'].unique().tolist()

    # Check wether file already exists, if yes, append to it
    if os.path.isfile(output_language_file):
        df_existing = pd.read_csv(os.path.join(output_folder,
                                               "caption_translation/en.txt"),
                                  sep='\t', names=["template", "unique_id", "caption", "caption_original"])
        common_rows = df_original.merge(
            df_existing[['unique_id']], on="unique_id", how='inner')
        df_original = df_original[~df_original['unique_id'].isin(
            common_rows['unique_id'])]

    all_captions = []
    for template in TEMPLATES:
        if CHOSEN_TEMPLATES and template.lower().strip() not in CHOSEN_TEMPLATES:
            continue
        print(f"Processing {template}")

        df_template = df_original[df_original["template_normalized"] == template].copy(
        )

        # Filter for English Languages
        df = filter_english(df_template)

        # Deduplicate
        print(len(df))
        df = df.drop_duplicates(subset='caption')
        print(len(df))

        # For now, randomly select:
        desired_sample_size = 300
        sample_size = min(len(df), desired_sample_size)
        if sample_size != desired_sample_size:
            print("Not Enough Samples for {}. Only {}.".format(
                template, sample_size))
        df = df.sample(n=sample_size)

        results = collect_results(df)
        all_captions += results

    if os.path.isfile(output_language_file):
        # Append to the text file
        with open(output_language_file, "a") as file:
            # Iterate through the list and write each string to the file
            for item in all_captions:
                # Adding a newline character after each string
                file.write(item)
    else:
        # Save into text file
        with open(output_language_file, "w") as file:
            # Iterate through the list and write each string to the file
            for item in all_captions:
                # Adding a newline character after each string
                file.write(item)
