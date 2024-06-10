import argparse
from google.cloud import translate_v3 as translate
from google.oauth2 import service_account
import pandas as pd
import os
from tqdm import tqdm

LANGUAGES = ["de"]
SIZE = 100
SEPERATOR = " // "

# Set up Google Translate credentials
project_id = 'fluted-karma-425309-j5'
credentials_path = '/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/fluted-karma-425309-j5-5c8d7c25596f.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
credentials = service_account.Credentials.from_service_account_file(
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
project_id = credentials.project_id
if project_id is None:
    raise Exception(
        "Could not determine the Google Cloud project ID from the service account key")
CLIENT = translate.TranslationServiceClient(credentials=credentials)
PARENT = f"projects/{project_id}/locations/global"


def translate_text(target: str, text: str) -> dict:
    response = CLIENT.translate_text(
        contents=[text],
        mime_type='text/plain',
        source_language_code='en',
        target_language_code=target,
        parent=PARENT,
    )

    translation = response.translations[0]
    result = {
        "translatedText": translation.translated_text,
    }
    return result


def translate_text_batch(target: str, texts: str) -> dict:
    response = CLIENT.translate_text(
        contents=texts,
        mime_type='text/plain',
        source_language_code='en',
        target_language_code=target,
        parent=PARENT,
    )

    results = [
        translation.translated_text for translation in response.translations]
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--generate_folder', '-g', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/generated_memes')
    parser.add_argument('--memes', '-m', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/memes')

    args = parser.parse_args()

    output_folder = args.generate_folder
    memes_folder = args.memes
    # Replace with your actual column names
    headers = ['template', 'instance_id', 'caption', 'caption_original']

    # Load the text file into a DataFrame
    df_selection = pd.read_csv(os.path.join(output_folder, "caption_translation/en.txt"),
                               sep='\t', names=headers)
    for language in LANGUAGES:
        all_captions = []

        # Define the batch size
        batch_size = 250

        # Iterate through the DataFrame in batches
        for start in tqdm(range(0, len(df_selection), batch_size), total=(len(df_selection) // batch_size + 1)):
            # Define the end of the current batch
            end = min(start + batch_size, len(df_selection))

            # Slice the DataFrame to get the current batch
            batch = df_selection.iloc[start:end]
            captions_original = list(batch['caption'])
            captions = [caption.replace(
                ' <sep> ', SEPERATOR) for caption in captions_original]
            translations = translate_text_batch(language, captions)
            for trans_index, translation in enumerate(translations):
                try:
                    top = captions_original[trans_index].split("<sep>")[
                        0].strip()
                    bottom = captions_original[trans_index].split("<sep>")[
                        1].strip()
                except IndexError as e:
                    print("Error:", e)
                    print("Error: Caption format incorrect for {}".format(
                        start+trans_index))
                if SEPERATOR not in translation:
                    top_trans = ""
                    bottom_trans = ""
                    if top != "":
                        top_trans = translate_text(language, top)[
                            "translatedText"]
                    if bottom != "":
                        bottom_trans = translate_text(language, bottom)[
                            "translatedText"]
                    text = top_trans + ' ' + "<sep>" + ' ' + bottom_trans
                else:
                    text = translation.replace(SEPERATOR, " <sep> ")

                link = batch["template"].iloc[trans_index]
                instance_id = batch["instance_id"].iloc[trans_index]
                text_original = batch["caption"].iloc[trans_index]
                all_captions.append(
                    f'{link}\t{instance_id}\t{text}\t{text_original}\n')

        # Create Path
        output_path = os.path.join(output_folder, "caption_translation")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_language_file = os.path.join(output_path, language + ".txt")

        # Save into text file
        with open(output_language_file, "w") as file:
            # Iterate through the list and write each string to the file
            for item in all_captions:
                # Adding a newline character after each string
                file.write(item)
