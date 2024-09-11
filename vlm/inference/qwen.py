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
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH = "/lustre/project/ki-topml/minbui/projects/models/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/3ca981c995b0ce691d85d8408216da11ff92f690"
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

    processed_prompts = []
    for prompt in PROMPTS:
        for postfix in PROMPT_POSTFIX:
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": "/lustre/project/ki-topml/minbui/projects/MultiModalHatespeech/hatespeech_nonhate/images/de/Advicejew/10402592.jpg"},
                    {"type": "text", "text": PROMPT_PREFIX + prompt + postfix},
                    ],
                },
            ]
            processed_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            processed_prompts.append([conversation, processed_prompt])

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype="auto", device_map="auto"
    )


    results_df = {"ID": [], "image_name": [], "prompt": [], "response": []}
    parent_dir = "/lustre/project/ki-topml/minbui/projects/MultiModalHatespeech/hatespeech_nonhate/images/en"

    model_postfix = MODEL_PATH.split("/")[-3]
    output_folder = os.path.join(args.output_folder, model_postfix)
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"responses_{args.language}.csv")
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
                        image_inputs, video_inputs = process_vision_info(prompt[0])
                        inputs = processor(
                            text=[prompt[1]],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt",
                        )
                        inputs = inputs.to("cuda")
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