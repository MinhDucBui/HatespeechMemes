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
MAX_CHAR_LENGTH = 100
desired_sample_size = 1200

CHOSEN_TEMPLATES = ["Scumbag-Catholic-Priest",
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

                    # Ethnicity
                    # "Provincial man",
                    # "Sassy Black Woman",
                    # "Native-American",

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
                    "Ignorant-White-Girl",
                    "Nice-White-Girl",
                    "White-Power-Problems",

                    # Race diversity
                    "Slavery",
                    "Privilege-Abusing-White-Couple",
                    "Privilege-Denying-Dude",

                    # Immigration
                    "Muslim-Immigrant",

                    # gender and family
                    "feminist cunt",
                    "Privilege-Denying-Feminist",
                    "Feministfrequently",

                    # Lesbian
                    "Lesbian-Scissor",
                    "Chinese-Lesbian",

                    # Trans
                    "Oppressive-Trans-Bro",
                    "Privilege-Denying-Tranny",
                    "Transvestite-Trevor",

                    # Gay
                    "Gay-Pornstar-Logic",
                    "Gay-Pride-Queer",
                    "Gay-Richard-Simmons",

                    # Police Man
                    "Scumbag-Police-Officer", 
                    "Strict-Policeman",
                    "Policeman"
                    ]


#CHOSEN_TEMPLATES = ["Provincial man"]
print("Number of Templates: {}".format(len(CHOSEN_TEMPLATES)))

CHOSEN_TEMPLATES = [template.replace("-", " ").lower().strip()
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

    if "900k" in folder_path:
        existing = args.generate_folder
    else:
        existing = None
        #CHOSEN_TEMPLATES = None

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
        df_original['unique_id'] = range(-1, -len(df_original) - 1, -1)
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
    count_template = 0
    count_captions = 0
    for template in TEMPLATES:
        template_check = template.replace("-", " ").lower().strip()
        if CHOSEN_TEMPLATES and template_check not in CHOSEN_TEMPLATES:
            continue
        print(f"\n------Processing {template}-------")

        df_template = df_original[df_original["template_normalized"] == template].copy(
        )

        # Filter for English Languages
        df = filter_english(df_template)

        # Deduplicate
        df = df.drop_duplicates(subset='caption')

        # Define the maximum length for captions
        # Filter rows where the caption length is within the allowed maximum length
        # Split the 'caption' into 'top' and 'bottom' parts using the separator '<sep>'
        df[['top_text', 'bottom_text']] = df['caption'].str.split(' <sep> ', expand=True)

        # Fill NaN values with empty strings if any part is missing
        df['top_text'] = df['top_text'].fillna('')
        df['bottom_text'] = df['bottom_text'].fillna('')

        # Filter rows where both the 'top_text' and 'bottom_text' lengths are within the allowed maximum lengths
        filtered_df = df[(df['top_text'].apply(len) <= MAX_CHAR_LENGTH) & 
                         (df['bottom_text'].apply(len) <= MAX_CHAR_LENGTH)]


        # For now, randomly select:
        sample_size = min(len(df), desired_sample_size)
        if sample_size != desired_sample_size:
            print("Not Enough Samples for {}. Only {}.".format(
                template, sample_size))
        df = df.sample(n=sample_size)

        results = collect_results(df)
        all_captions += results
        count_template += 1

    print(f"Processed {count_template} Templates")

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
