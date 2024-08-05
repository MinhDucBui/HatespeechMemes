import argparse
from google.cloud import translate_v3 as translate
from google.oauth2 import service_account
import pandas as pd
import os
from tqdm import tqdm
from PIL import Image


LANGUAGES = ["en", "de", "es", "hi", "zh"]
SIZE = 100
SAMPLE_SIZE = 3
SEPERATOR = " // "

# Set up Google Translate credentials
project_id = 'marine-booth-428514-a9'
credentials_path = '/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/marine-booth-428514-a9-6d11270fe246.json'
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


def process_df_wordcheck(df, sample_size=SAMPLE_SIZE):
    df = df[df["translation_check"] == 1]
    df_group = df.groupby('processed_template')
    # Check for groups with fewer than 10 rows and raise an error if any are found
    if (group_sizes := df_group.size()).min() < sample_size:
        raise ValueError(
            f"Some groups have fewer than {SAMPLE_SIZE} rows: {group_sizes[group_sizes < sample_size].to_dict()}")

    df = df_group.apply(lambda x: x[:sample_size])
    df = df.reset_index(drop=True)
    return df


def resize_image(filename, new_width, new_height):
    # Open the image
    img = Image.open(filename)

    # Resize the image
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return resized_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--wordplay_check', '-w', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/checking_wordplay_final')
    parser.add_argument('--output', '-o', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/translation_nonhate2')
    parser.add_argument('--generate_folder', '-g', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/generated_memes')
    parser.add_argument('--memes', '-m', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/memes')
    args = parser.parse_args()

    output_folder = args.output
    wordplay_check_folder = args.wordplay_check
    generate_folder = args.generate_folder
    folder_path = args.memes

    # Replace with your actual column names
    headers = ['template', 'instance_id', 'caption', 'caption_original']

    for language in LANGUAGES:
        # Load the text file into a DataFrame
        df_selection1 = pd.read_excel(os.path.join(
            wordplay_check_folder, "nonhate_doing.xlsx"))
        df_selection1 = process_df_wordcheck(df_selection1)
        # df_selection2 = pd.read_excel(os.path.join(
        #     wordplay_check_folder, "random.xlsx"))
        # df_selection2 = process_df_wordcheck(df_selection2)
        df_selection = df_selection1
        df_selection = df_selection.sort_values(by=['processed_template'])
        all_translations = []

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

            if language == "en":
                translations = captions
            else:
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

                all_translations.append(text)

        # Create Path
        output_path = os.path.join(output_folder, "raw")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_language_file = os.path.join(
            output_path, language + "_translation" + ".xlsx")

        df_selection["translation"] = all_translations
        df_selection = df_selection[[
            "instance_id", "processed_template", "caption", "translation"]]
        # Rename columns
        df_selection = df_selection.rename(columns={
            'instance_id': 'ID',
            'processed_template': 'Template Name',
            'caption': 'Original (English)',
            'translation': 'Translation'
        })
        df_selection["Correct (=1) or False (=0)"] = pd.NA
        df_selection["Correct Translation"] = pd.NA

        # Create empty row
        unique_templates = df_selection['Template Name'].unique()
        new_rows = []
        for template in unique_templates:
            new_rows.append(
                {'ID': '', 'Template Name': '', 'Original (English)': "", "Translation": ""})
            rows_to_add = df_selection[df_selection['Template Name'] == template].to_dict(
                'records')
            new_rows.extend(rows_to_add)
        df_selection = pd.DataFrame(new_rows)

        # Save the DataFrame to an Excel file
        df_selection.to_excel(output_language_file,
                              sheet_name='Sampled Data', index=False)

        df_caption = pd.read_csv(os.path.join(
            generate_folder, "caption_translation", "en.txt"), sep='\t', names=headers)
        empty_id_rows = df_selection[df_selection['ID'] == ""]

        writer = pd.ExcelWriter(output_language_file, engine='xlsxwriter')
        df_selection.to_excel(writer, sheet_name='Sheet1')
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        worksheet.set_default_row(100)
        col_idx = df_selection.columns.get_loc('ID') + 1
        writer.sheets['Sheet1'].set_column(col_idx, col_idx, 20)
        for index, row in empty_id_rows.iterrows():
            instance_id = df_selection["ID"].iloc[index+1]
            # Get Image Path
            template_name = df_caption[df_caption["instance_id"]
                                       == instance_id]["template"].iloc[0]
            template_name = template_name.replace(" ", "-")
            image_file = os.path.join(
                folder_path, "images", template_name + ".jpg")

            if not os.path.exists(image_file):
                image_file = os.path.join(
                    folder_path + "900k", "images", template_name + ".jpg")
            new_width = 100
            new_height = 100
            img = resize_image(image_file, new_width, new_height)
            directory, file_name = os.path.split(image_file)
            new_file_name = f"resized_{file_name}"
            image_file = os.path.join(directory, new_file_name)
            img.save(image_file, 'JPEG')
            worksheet.insert_image('B' + str(index+2),  image_file)
        writer.book.close()
