import argparse
import pandas as pd
import os
from google.cloud import translate_v3 as translate
from google.oauth2 import service_account


SAMPLE_SIZE = 15

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

CHOSEN_TEMPLATES = [
    "Scumbag-Catholic-Priest",
    "Condescending-Christian",
    "Jesus-Christ",

    # Islam
    "Angry Muslim Guy",
    "Ordinary-Muslim-Man",
    "Confused-Muslim-Girl",

    # Judaism
    "Advicejew",
    "jewish-dude",
    "like-a-jew",

    # Germany
    "Typical-Germany-Lover",
    "Germany-Pls",
    "Success-Germany",

    # USA
    "American Pride Eagle",
    "American-Flag-Shotgun-Guy",
    "Obese-American",

    # Mexico
    "Successful Mexican",
    "Mexicanotriste",
    "Mexicans-On-A-Truck",

    # China
    "Stern-But-Honest-Chinese-Guy",
    "Good-Chinese-Student",
    "Nia-China",

    # India
    "Generic Indian guy",
    "Indian-Father",
    "Stereotypical-Indian-Telemarketer",

    # Asian
    "Asian-College-Freshman",
    "Bad-Advice-Asian",
    "Troll-Asian",

    # Black
    "Sassy Black Woman",
    "Black Kid",
    "Skeptical-Black-Kid",

    # Middle Eastern
    "Y-U-So-Arab",
    "Arabic-Meme",
    "Richarabclap",

    # White
    "Privilege-Abusing-White-Couple",
    "Nice-White-Girl",
    "White-Power-Problems",

    # Race diversity
    # "Slavery",
    # "Privilege-Abusing-White-Couple",
    # "Privilege-Denying-Dude",

    # Immigration
    "Muslim-Immigrant",

    # Lesbian
    # "Lesbian-Scissor",
    # "Chinese-Lesbian",

    # Trans
    "Oppressive-Trans-Bro",
    "Privilege-Denying-Tranny",
    "Transvestite-Trevor",

    # Gay
    # "Gay-Pornstar-Logic",
    # "Gay-Pride-Queer",
    # "Gay-Richard-Simmons",

    # gender and family
    "feminist cunt",
    "Privilege-Denying-Feminist",
    "Feministfrequently",

    # Police Man
    "Scumbag-Police-Officer",
    "Strict-Policeman",
    "Policeman"
]

CHOSEN_TEMPLATES = [template.replace("-", " ").lower().strip()
                    for template in CHOSEN_TEMPLATES]
SEPERATOR = " // "


def translate_text(caption: str, target="de") -> dict:
    response = CLIENT.translate_text(
        contents=caption,
        mime_type='text/plain',
        source_language_code='en',
        target_language_code=target,
        parent=PARENT,
    )

    translation = [text.translated_text for text in response.translations]
    return translation


def batch_translate(captions, batch_size=2):
    """
    Translate the text in batches.

    :param df: DataFrame containing the text to be translated.
    :param column: Column name in DataFrame that contains the text.
    :param batch_size: Number of rows to process in each batch.
    :return: List of translated texts.
    """
    translations = []

    # Iterate over the DataFrame in batches
    for start in range(0, len(captions), batch_size):
        end = start + batch_size
        batch = captions[start:end]
        # Call the translation function
        translated_batch = translate_text(batch)
        translations.extend(translated_batch)

    return translations


def translate_dataframe(df_subset):
    captions_original = list(df_subset['caption'])
    captions = [caption.replace(
        ' <sep> ', SEPERATOR) for caption in captions_original]
    translations = batch_translate(captions, batch_size=400)
    translations_postprocessed = []
    for trans_index, translation in enumerate(translations):
        top = captions_original[trans_index].split("<sep>")[
            0].strip()
        bottom = captions_original[trans_index].split("<sep>")[
            1].strip()
        if SEPERATOR not in translation:
            top_trans = ""
            bottom_trans = ""
            if top != "":
                top_trans = translate_text([top], "de")[0]
            if bottom != "":
                bottom_trans = translate_text([bottom], "de")[0]
            text = top_trans + ' ' + "<sep>" + ' ' + bottom_trans
        else:
            text = translation.replace(SEPERATOR, " <sep> ")
        translations_postprocessed.append(text)
    df_subset['translation'] = translations_postprocessed
    return df_subset


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--results_file', '-r', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/generated_memes/results/results.xlsx')
    parser.add_argument('--final_dataset_path', '-f', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/asd')

    args = parser.parse_args()
    results_file = args.results_file
    final_dataset_path = args.final_dataset_path

    df = pd.read_excel(results_file)

    templates = list(set(list(df["processed_template"])))
    templates = sorted(templates)
    df["translation_check"] = [-1] * len(df)
    df["score"] = [-1] * len(df)

    df_collect_multimodal = []
    df_collect_false = []
    for template in templates:
        template_check = template.replace("-", " ").lower().strip()
        if template_check not in CHOSEN_TEMPLATES:
            continue
        print(f"\n-----Processing Template: {template}----------")
        df_subset = df[df["processed_template"] == template]
        print(f"Total Length Original: {len(df_subset)}")

        # Deduplicate
        df_subset = df_subset.drop_duplicates(subset=['caption'])

        # Differences
        df_true = df_subset[df_subset["difference"] == True]
        df_any = df_subset[(df_subset["difference_any"] == True)
                           & (df_subset["difference"] == False)]
        df_false = df_subset
        print(f"Total True Length: {len(df_true)}")
        print(f"Total Any Length: {len(df_any)}")

        # df: True
        template_path = os.path.join(final_dataset_path, template)
        if not os.path.exists(template_path):
            os.makedirs(template_path)
        multimodal_hate_path = os.path.join(template_path, "multimodal.xlsx")
        if SAMPLE_SIZE <= len(df_true):
            df_true = df_true.sample(n=SAMPLE_SIZE)
        else:
            # Soft Threshold
            df_any = df_any.sample(frac=1).reset_index(drop=True)
            df_true = pd.concat([df_true, df_any])
            if SAMPLE_SIZE <= len(df_true):
                df_true = df_true[:SAMPLE_SIZE]
            else:
                # No Threshold
                df_false = df_false.sample(frac=1).reset_index(drop=True)
                df_true = pd.concat([df_true, df_false])
                df_true = df_true[:SAMPLE_SIZE]

        df_true = translate_dataframe(df_true)
        df_true.to_excel(multimodal_hate_path, index=False)

        # Random ones
        df_false = df_false[~df_false['instance_id'].isin(
            df_true['instance_id'])]
        sample_size = min(len(df_false), SAMPLE_SIZE)
        if sample_size != 0:
            multimodal_hate_path = os.path.join(template_path, "random.xlsx")
            df_false = df_false.sample(n=sample_size)
            df_false = translate_dataframe(df_false)
            df_false.to_excel(multimodal_hate_path, index=False)

        df_collect_multimodal.append(df_true)
        df_collect_false.append(df_false)

    multimodal_hate_path = os.path.join(final_dataset_path, "multimodal.xlsx")
    df_collect_multimodal = pd.concat(df_collect_multimodal)
    df_collect_multimodal.to_excel(multimodal_hate_path, index=False)

    multimodal_hate_path = os.path.join(final_dataset_path, "random.xlsx")
    df_collect_false = pd.concat(df_collect_false)
    df_collect_false.to_excel(multimodal_hate_path, index=False)
