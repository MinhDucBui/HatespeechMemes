import requests
from PIL import Image
import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
two_dirs_up = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
sys.path.append(two_dirs_up)

from annotation_evaluation.utils import process_language_data
from tqdm import tqdm
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import argparse
import pandas as pd
import random


MODEL_PATH = "/lustre/project/ki-topml/minbui/projects/models/models--llava-hf--llava-1.5-7b-hf/snapshots/95f2cf316249a98c11183f471ab76065c3e5f494"
# MODEL_PATH = "/lustre/project/ki-topml/minbui/projects/models/models--llava-hf--llava-v1.6-vicuna-7b-hf/snapshots/89b0f2ea28da2e62d7cfda173a400d2ad82a1c8e"
IMAGE_FOLDER = "/lustre/project/ki-topml/minbui/projects/MultiModalHatespeech/hatespeech_nonhate/images/"


PROMPT_PREFIX = "Definition of Hate Speech:\n\nAny kind of communication [...], that attacks or uses pejorative or discriminatory language with reference to a person or a group on the basis of who they are, in other words, based on their religion, ethnicity, nationality, race, colour, descent, gender or other identity factor.\n\n"
PROMPT_POSTFIX = ["\n\na: Hate\nb: Non-hate\nanswer:", "\n\na: Non-hate\nb: Hate"]

PROMPTS = [
    "Answer if this meme is hate or not with a single alphabet letter among given answer choices a and b.",
    "Given a meme from online, answer if it is 'Hate Speech' or 'Non-Hate Speech'. Answer with only one single alphabet between 'a' and 'b', with no other description.",
    "Would you consider this meme as a hate speech? Give an answer in a single alphabet only.",
    "Do you think this is a hate speech? Answer with only one single alphabet between 'a' and 'b', with no other description.",
    "Would you consider this meme as a hate speech? Give an answer in a single alphabet only.",
]

def get_device_map() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--language', '-l', type=str, default='en')
    parser.add_argument('--annotation', '-a', type=str, default='/lustre/project/ki-topml/minbui/projects/MultiModalHatespeech/prolific/hatespeech_main/')
    parser.add_argument('--output_folder', '-o', type=str, default='/lustre/project/ki-topml/minbui/projects/MultiModalHatespeech/model_predictions')
    args = parser.parse_args()

    device = get_device_map()
    df = process_language_data(args.annotation)

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model_postfix = MODEL_PATH.split("/")[-3]
    output_folder = os.path.join(args.output_folder, model_postfix)
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"responses_{args.language}.csv")
    processed_prompts = []
    for prompt in PROMPTS:
        for postfix in PROMPT_POSTFIX:
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_PREFIX + prompt + postfix},
                    {"type": "image"},
                    ],
                },
            ]
            processed_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            processed_prompts.append(processed_prompt)

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        device_map=device 
    )


    results_df = {"ID": [], "image_name": [], "prompt": [], "response": []}
    parent_dir = "/lustre/project/ki-topml/minbui/projects/MultiModalHatespeech/hatespeech_nonhate/images/en"
    for root, dirs, files in tqdm(os.walk(parent_dir), total=45):
        for file in files:
            # Check if the file is an image by its extension
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                image_path = os.path.join(root, file)
                image_path_check = str(image_path.split("/")[-1].split(".")[0])
                # print(list(df.index))
                if image_path_check in list(df.index):
                    raw_image = Image.open(image_path)
                    for index_prompt, prompt in enumerate(processed_prompts):
                        inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
                        output = model.generate(**inputs, max_new_tokens=10, do_sample=False, temperature=1.0)
                        response_text = processor.decode(output[0][2:], skip_special_tokens=True)
                        results_df["ID"].append(image_path_check)
                        results_df["prompt"].append(index_prompt)
                        results_df["image_name"].append(image_path)
                        results_df["response"].append(response_text)

        save_df = pd.DataFrame(results_df)
        pd.set_option('display.max_colwidth', None)
        print(save_df.iloc[-2:]["response"])
        save_df.to_csv(output_file, index=False)

    save_df = pd.DataFrame(results_df)
    output_file = f"responses_{args.language}.csv"
    save_df.to_csv(output_file, index=False)