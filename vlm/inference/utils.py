import pandas as pd
import torch
from tqdm import tqdm
import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
two_dirs_up = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
sys.path.append(two_dirs_up)

from annotation_evaluation.utils import process_language_data


IMAGE_FOLDER = "/lustre/project/ki-topml/minbui/projects/MultiModalHatespeech/hatespeech_nonhate/images/"
PROMPT_PREFIX = "Definition of Hate Speech:\n\nAny kind of communication [...], that attacks or uses pejorative or discriminatory language with reference to a person or a group on the basis of who they are, in other words, based on their religion, ethnicity, nationality, race, colour, descent, gender or other identity factor.\n\n"
PROMPT_POSTFIX = ["\n\na: Hate\nb: Non-hate\n", "\n\na: Non-hate\nb: Hate"]
PROMPTS = [
    "Answer if this meme is hate or not with a single alphabet letter among given answer choices a and b.",
    "Given a meme from online, answer if it is 'Hate Speech' or 'Non-Hate Speech'. Answer with only one single alphabet between 'a' and 'b', with no other description.",
    "Would you consider this meme as a hate speech? Give an answer in a single alphabet only.",
    "Do you think this is a hate speech? Answer with only one single alphabet between 'a' and 'b', with no other description.",
    "Would you consider this meme as a hate speech? Give an answer in a single alphabet only.",
]


def get_device_map() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def name_output_file(model_path, output_folder, language):
    model_postfix = model_path.split("/")[-3]
    output_folder = os.path.join(output_folder, model_postfix)
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"responses_{language}.csv")
    return output_file


def pipeline_inference(model_path, language, annotation_path, output_folder, input_creator, model_creator, model_inference):
    # Image list
    image_paths = []
    results_df = {"ID": [], "image_name": [], "prompt": [], "response": []}
    parent_dir = "/lustre/project/ki-topml/minbui/projects/MultiModalHatespeech/hatespeech_nonhate/images/" + language
    df = process_language_data(annotation_path)

    for root, _, files in tqdm(os.walk(parent_dir), total=45):
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
            all_prompts.append(PROMPT_PREFIX + prompt + postfix)

    # Prompt Creation
    processor, processed_inputs = input_creator(
        all_prompts, image_paths, model_path)

    # Model Creation
    model = model_creator()

    # Main Inference Loop
    results_df = {"ID": [], "image_name": [], "prompt": [], "response": []}
    for idx, (model_input, image_path) in enumerate(zip(processed_inputs, image_paths)):
        model_input["model"] = model
        model_input["processor"] = processor
        response_text = model_inference(**model_input)

        # Collect
        id_image = str(image_path.split("/")[-1].split(".")[0])
        results_df["ID"].append(id_image)
        index_prompt = idx % 10
        results_df["prompt"].append(index_prompt)
        results_df["response"].append(response_text)

    save_df = pd.DataFrame(results_df)
    output_file = name_output_file(model_path, output_folder, language)
    save_df.to_csv(output_file, index=False)
