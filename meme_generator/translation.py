import argparse
from googletrans import Translator
import pandas as pd
import os
from tqdm import tqdm

LANGUAGES = ["de"]

def translation(topString, bottomString, language):
    translator = Translator()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--memes', '-m', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/memes')
    parser.add_argument('--generate_folder', '-g', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/generated_memes')
    parser.add_argument('--crawled', '-c', required=True, type=str,
                         help='directory where the dataset should be stored')
    args = parser.parse_args()

    folder_path = args.memes
    output_folder = args.generate_folder

    headers = ['template', 'id', 'caption']  # Replace with your actual column names

    # Load the text file into a DataFrame
    df = pd.read_csv(os.path.join(folder_path, "captions.txt"), sep='\t', names=headers)
    df = df.drop_duplicates()

    translator = Translator()
    for language in LANGUAGES:
        all_captions = []
        for index, row in tqdm(df.iterrows()):
            print(row)
            # Get Bottom & Top Strings
            top = row["caption"].split("<sep>")[0].strip()
            bottom = row["caption"].split("<sep>")[1].strip()

            # Google Translation
            if top != "":
                top_trans = translator.translate(top, src='en', dest="de").text
            if bottom != "":
                bottom_trans = translator.translate(bottom, src='en', dest="de").text
            text = top_trans + ' ' + "<sep>" + ' ' + bottom_trans
            link = row["template"]
            instance_id = row["instance_id"]
            src = row["img_link"]
            image_path = row["img"]
            all_captions.append(f'{link}\t{instance_id}\t{text}\t{src}\t{image_path}\n')

        # Create Path
        output_path = os.path.join(output_folder, "caption_translation")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_language_file = os.path.join(output_path, language + ".txt")

        # Save into text file
        with open(output_language_file, "w") as file:
            # Iterate through the list and write each string to the file
            for item in all_captions:
                file.write(item)  # Adding a newline character after each string
