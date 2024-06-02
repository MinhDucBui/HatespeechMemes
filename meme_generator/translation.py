import argparse
from googletrans import Translator
import pandas as pd
import os
from tqdm import tqdm

LANGUAGES = ["de"]
SIZE = 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--generate_folder', '-g', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/generated_memes')
    parser.add_argument('--memes', '-m', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/memes')

    args = parser.parse_args()

    output_folder = args.generate_folder
    memes_folder = args.memes
    headers = ['template', 'instance_id', 'caption', 'caption_original']  # Replace with your actual column names

    # Load the text file into a DataFrame
    df_selection = pd.read_csv(os.path.join(output_folder, "caption_translation/en.txt"),
                               sep='\t', names=headers)

    translator = Translator()

    for language in LANGUAGES:
        all_captions = []
        for index, row in tqdm(df_selection.iterrows()):
            # Get Bottom & Top Strings
            try:
                top = row["caption"].split("<sep>")[0].strip()
                bottom = row["caption"].split("<sep>")[1].strip()
                text_original = top + ' ' + "<sep>" + ' ' + bottom
            except IndexError as e:
                print("Error:", e)
                print("Error: Caption format incorrect for {}".format(index))

            # Google Translation
            SEPERATOR = " // "
            text = top + SEPERATOR + bottom
            if language != "en":
                text = translator.translate(text, src='en', dest=language).text

            # Postprocessing of seperator
            # text = text.replace("<Schritt>", "<sep>")
            # text = text.replace("<Sep>", "<sep>")
            # text = text.replace("<step>", "<sep>")
            if SEPERATOR not in text:
                # Then have to translate top and bottom separately
                top_trans = ""
                bottom_trans = ""
                if top != "":
                    top_trans = translator.translate(top, src='en', dest="de").text
                if bottom != "":
                    bottom_trans = translator.translate(bottom, src='en', dest="de").text
                text = top_trans + ' ' + "<sep>" + ' ' + bottom_trans
            else:
                text = text.replace(SEPERATOR, " <sep> ")

            link = row["template"]
            instance_id = row["instance_id"]
            all_captions.append(f'{link}\t{instance_id}\t{text}\t{text_original}\n')

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
