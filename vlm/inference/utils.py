import pandas as pd
import torch
from tqdm import tqdm
import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
two_dirs_up = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
sys.path.append(two_dirs_up)

from annotation_evaluation.utils import process_language_data
from vlm.inference.local_paths import ANNOTATION_PATH, OUTPUT_FOLDER, IMAGE_FOLDER, CAPTION_FOLDER
from vlm.inference.all_prompts import set_prompts

# Caption
UNIMODAL = False
PREFIX = "new-"

# Prompt Definition
PROMPT_PREFIX = "Definition of Hate Speech:\n\nAny kind of communication [...], that attacks or uses pejorative or discriminatory language with reference to a person or a group on the basis of who they are, in other words, based on their religion, ethnicity, nationality, race, colour, descent, gender or other identity factor.\n\n"
PROMPT_POSTFIX = ["\nChoose:\na: Hate\nb: Non-hate\n", "\nChoose:\na: Non-hate\nb: Hate\n"]
PROMPTS = [
    "Answer if this meme is hate or not with a single alphabet letter among given answer choices a and b.",
    "Given a meme from online, answer if it is 'Hate Speech' or 'Non-Hate Speech'. Answer with only one single alphabet between 'a' and 'b', with no other description.",
    "Would you consider this meme as a hate speech? Give an answer in a single alphabet only.",
]
PROMPTS_COUNTRY_INSERTION = [
    "Answer if this meme is hate or not in {} with a single alphabet letter among given answer choices a and b.",
    "Given a meme from online, answer if it is 'Hate Speech' or 'Non-Hate Speech' in {}. Answer with only one single alphabet between 'a' and 'b', with no other description.",
    "As a person raised up in {}, would you consider this meme as a hate speech? Give an answer in a single alphabet only.",
]
PROMPT_NUMBER = len(PROMPTS) * len(PROMPT_POSTFIX)
PROMPT_IMAGE_PREFIX = "\nMeme: "
PROMPT_CAPTION = "\nCaption inside the meme: '{}'"
MAPPING = {
    "en": "the United States",
    "de": "Germany",
    "es": "Mexico",
    "hi": "India",
    "zh": "China", 
}


def create_prompt_for_input(raw_prompt, df_captions, image_path, add_caption):
    prompt_1 = raw_prompt[0]
    prompt_2 = raw_prompt[1]
    if add_caption:
        id_image = image_path.split("/")[-1].split(".jpg")[0]
        caption = df_captions[df_captions["ID"]
                                == id_image]["Translation"].iloc[0]

        text_prompt_1 = {"type": "text", "text": prompt_1}
        text_prompt_2 = {"type": "text", "text": prompt_2.format(str(caption))}
    else:
        text_prompt_1 = {"type": "text", "text": prompt_1}
        text_prompt_2 = {"type": "text", "text": prompt_2}

    return text_prompt_1, text_prompt_2


def get_device_map() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def name_output_file(model_path, output_folder, language, add_caption):
    model_postfix = model_path.split("/")[-3]
    if add_caption:
        model_postfix = model_postfix + "_caption"
    if PREFIX:
        model_postfix = PREFIX + model_postfix
    output_folder = os.path.join(output_folder, model_postfix)
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"responses_{language}.csv")
    return output_file

def process_translations(final_dataset, language):
    # Load the text file into a DataFrame
    final_file = os.path.join(
        final_dataset, language + "_translation.xlsx")
    df_annotation = pd.read_excel(final_file)
    df_annotation = df_annotation.dropna(subset=['ID'])
    # df_annotation = df_annotation[df_annotation["ID"] == -329162.0]

    # Remove zero-width space character (\u200b)
    df_annotation['Translation'] = df_annotation['Translation'].apply(
        lambda x: x if pd.isna(x) else x.replace('\u200b', ''))
    df_annotation['Correct Translation'] = df_annotation['Correct Translation'].apply(
        lambda x: x if pd.isna(x) else x.replace('\u200b', ''))
    # df_annotation = df_annotation[df_annotation['ID'].isin([56603606, 14106131, 11807098, 44355237])]
    # df_annotation = df_annotation[df_annotation['ID'].isin([72039904])]
    df_annotation = df_annotation.drop_duplicates()
    df_annotation = df_annotation.sample(frac=1).reset_index(drop=True)

    # Create a mask for rows where 'Correct Translation' is not empty
    non_empty = pd.notna(df_annotation['Correct Translation'])
    df_annotation.loc[non_empty,
                      'Translation'] = df_annotation.loc[non_empty, 'Correct Translation']
    if "2nd: Correct Translation" in df_annotation.keys():
        non_empty = pd.notna(df_annotation['2nd: Correct Translation'])
        df_annotation.loc[non_empty, 'Translation'] = df_annotation.loc[non_empty,
                                                                        '2nd: Correct Translation']

    if "3nd: Correct Translation" in df_annotation.keys():
        non_empty = pd.notna(df_annotation['3nd: Correct Translation'])
        df_annotation.loc[non_empty, 'Translation'] = df_annotation.loc[non_empty,
                                                                        '3nd: Correct Translation']
    df_annotation = df_annotation[['ID', 'Template Name', 'Original (English)', 'Translation']]
    return df_annotation


def pipeline_inference(model_path, languages, input_creator, model_creator, model_inference, add_caption=False, multilingual=False, country_insertion=False):
    global PROMPTS, PROMPT_CAPTION, PROMPT_PREFIX, PROMPT_POSTFIX, PROMPT_IMAGE_PREFIX, PREFIX
    # Model Creation
    model = model_creator(model_path)

    MULTILINGUAL = multilingual
    COUNTRY_INSERTION = country_insertion
    if MULTILINGUAL:
        PREFIX = "multilingual_" + PREFIX

    if COUNTRY_INSERTION:
        PREFIX = "country_insertion_" + PREFIX

    if UNIMODAL:
        PREFIX = "unimodal_" + PREFIX

    for language in languages:
        print("\n-----Processing {} Language\n".format(language))
        if MULTILINGUAL:
            PROMPTS, PROMPT_CAPTION, PROMPT_PREFIX, PROMPT_POSTFIX, PROMPT_IMAGE_PREFIX = set_prompts(language)
        if COUNTRY_INSERTION:
            PROMPTS = PROMPTS_COUNTRY_INSERTION
        # Load Captions
        df_captions = process_translations(CAPTION_FOLDER, language)
        df_captions["ID"] = df_captions["ID"].astype(int).astype(str)

        # Image list
        image_paths = []
        results_df = {"ID": [], "image_name": [], "prompt": [], "response": []}
        parent_dir = IMAGE_FOLDER + language
        df = process_language_data(ANNOTATION_PATH)

        for root, _, files in os.walk(parent_dir):
            for file in files:
                # Check if the file is an image by its extension
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                    image_path = os.path.join(root, file)
                    image_path_check = str(image_path.split("/")[-1].split(".")[0])
                    if image_path_check in list(df.index):
                        image_paths.append(image_path)

        # All prompts
        all_prompts = []
        for prompt in PROMPTS:
            for postfix in PROMPT_POSTFIX:
                prompt_1 = PROMPT_PREFIX + prompt + PROMPT_IMAGE_PREFIX
                if COUNTRY_INSERTION:
                    prompt_1 = prompt_1.format(MAPPING[language])
                if add_caption:
                    all_prompts.append([prompt_1, PROMPT_CAPTION + postfix])
                else:
                    all_prompts.append([prompt_1, postfix])

        # Prompt Creation
        processor, processed_inputs = input_creator(
            all_prompts, image_paths, model_path, df_captions, add_caption=add_caption)

        # Main Inference Loop
        results_df = {"ID": [], "prompt": [], "response": []}
        image_paths = [item for item in image_paths for _ in range(PROMPT_NUMBER)]
        max_length = len(processed_inputs)
        for idx, (model_input, image_path) in tqdm(enumerate(zip(processed_inputs, image_paths)), total=max_length):
            #if idx < 200:
            #    continue

            model_input["model"] = model
            model_input["processor"] = processor
            response_text = model_inference(**model_input)

            # Collect
            id_image = str(image_path.split("/")[-1].split(".")[0])
            results_df["ID"].append(id_image)
            index_prompt = idx % PROMPT_NUMBER
            results_df["prompt"].append(index_prompt)
            results_df["response"].append(response_text)

            if idx % 100 == 0:
                save_df = pd.DataFrame(results_df)
                output_file = name_output_file(model_path, OUTPUT_FOLDER, language, add_caption)
                print(output_file)
                save_df.to_csv(output_file, index=False)

        save_df = pd.DataFrame(results_df)
        output_file = name_output_file(model_path, OUTPUT_FOLDER, language, add_caption)
        save_df.to_csv(output_file, index=False)
