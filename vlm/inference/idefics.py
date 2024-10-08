from transformers import IdeficsForVisionText2Text, AutoProcessor
import torch
from PIL import Image
import sys
import os
import argparse

current_script_dir = os.path.dirname(os.path.abspath(__file__))
two_dirs_up = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
sys.path.append(two_dirs_up)

from vlm.inference.utils import pipeline_inference, create_prompt_for_input

LANGUAGES = ["en", "de", "es", "hi", "zh"]
# MODEL_PATH = "/lustre/project/ki-topml/minbui/projects/models/models--llava-hf--llava-v1.6-34b-hf/snapshots/66b6feb83d0249dc9f31a24bd3abfb63f90e41aa"
# MODEL_PATH = "/lustre/project/ki-topml/minbui/projects/models/models--llava-hf--llava-v1.6-vicuna-13b-hf/snapshots/e66fcaaa7d502b1037c8465375bb67f4c33758dd"
# MODEL_PATH = "/lustre/project/ki-topml/minbui/projects/models/sync/models--llava-hf--llava-next-72b-hf/snapshots/834da453803d866c3b45f4f94dc20f5b705d5a88"
UNIMODAL = False

def input_creator(all_prompts, image_paths, model_path, df_captions, add_caption):
    # Input for model_inference()
    processor = AutoProcessor.from_pretrained(model_path)
    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids


    processor.patch_size = 14
    processor.vision_feature_select_strategy = "default"

    processed_prompts = []
    for image_path in image_paths:
        for raw_prompt in all_prompts:
            text_prompt_1, text_prompt_2 = create_prompt_for_input(raw_prompt, df_captions, image_path, add_caption)
            
            raw_image = Image.open(image_path)
            prompts = [
                [
                    "User: " + "Describe the image:",
                    raw_image,
                    "First, start with answer:",
                    "<end_of_utterance>",
                    "\nAssistant:",
                ],
            ]
            processed_prompts.append(
                {"prompt": [prompts, exit_condition, bad_words_ids], "image_path": image_path})

    return processor, processed_prompts


def model_creator(model_path):
    model = IdeficsForVisionText2Text.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    return model


def model_inference(image_path, prompt, model, processor):
    inputs = processor(prompt[0], add_end_of_utterance_token=False, return_tensors="pt").to("cuda")
    print("start")

    generated_ids = model.generate(**inputs, max_new_tokens=20)
    print("end")
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

    response_text = ""
    for i, t in enumerate(generated_text):
        response_text = response_text + f"{i}:\n{t}\n"

    return response_text


if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Run pipeline inference with specified model path.')

    # Add an argument for MODEL_PATH
    parser.add_argument('--model_path', type=str, required=False, default='/lustre/project/ki-topml/minbui/projects/models/models--HuggingFaceM4--idefics-80b-instruct/snapshots/a14d258b1be2a74a3604483de552c33121a98391')
    parser.add_argument('--caption', action='store_true', help='Enable captioning')
    args = parser.parse_args()

    pipeline_inference(args.model_path, LANGUAGES, input_creator, model_creator, model_inference, add_caption=args.caption)
