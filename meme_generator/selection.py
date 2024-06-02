import argparse
from googletrans import Translator
import pandas as pd
import os
from tqdm import tqdm
from langdetect import detect
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

LANGUAGES = ["de"]
SIZE = 100


def filter_english(df):
    df['caption_modified'] = df['caption'].str.replace('<sep>', '')
    df['caption_modified'] = df['caption_modified'].str.replace('<emp>', '')
    df['caption_modified'] = df['caption_modified'].str.replace('  ', ' ')
    df['caption_modified'] = df['caption_modified'].str.lower()
    tqdm.pandas()
    df['en_detect'] = df['caption_modified'].progress_apply(detect)

    df = df[df['en_detect'] == "en"]
    #print("New Length: {}".format(len(df)))
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--memes', '-m', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/memes')
    parser.add_argument('--generate_folder', '-g', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/generated_memes')
    # parser.add_argument('--save-dir', '-d', required=True, type=str,
    #                    help='directory where the dataset should be stored')
    args = parser.parse_args()

    folder_path = args.memes
    output_folder = args.generate_folder
    headers = ['template', 'instance_id', 'caption']  # Replace with your actual column names

    # Load the text file into a DataFrame
    df_original = pd.read_csv(os.path.join(folder_path, "captions.txt"), sep='\t', names=headers)
    df_original['unique_id'] = range(1, len(df_original) + 1)
    df_original = df_original.drop_duplicates()

    translator = Translator()

    df_original['template_normalized'] = df_original['template'].str.split('_variant=').str[0]
    TEMPLATES = df_original['template_normalized'].unique().tolist()

    for template in TEMPLATES:
        df_template = df_original[df_original["template_normalized"] == template].copy()

        # Filter for English Languages
        df = filter_english(df_template)

        # For now, randomly select:
        df = df.sample(n=100)

        # Save Selection
        all_captions = []
        for index, row in df.iterrows():
            link = row["template"]
            instance_id = row["unique_id"]
            top = row["caption"].split("<sep>")[0].strip()
            bottom = row["caption"].split("<sep>")[1].strip()
            text_original = top + ' ' + "<sep>" + ' ' + bottom
            all_captions.append(f'{link}\t{instance_id}\t{text_original}\t{text_original}\n')

        # Create Path
        output_path = os.path.join(output_folder, "caption_translation")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_language_file = os.path.join(output_path, "en.txt")

        # Save into text file
        with open(output_language_file, "w") as file:
            # Iterate through the list and write each string to the file
            for item in all_captions:
                file.write(item)  # Adding a newline character after each string
