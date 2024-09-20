from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import sys
import os
import argparse

current_script_dir = os.path.dirname(os.path.abspath(__file__))
two_dirs_up = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
sys.path.append(two_dirs_up)

from vlm.inference.utils import pipeline_inference

LANGUAGES = ["es", "hi"]
MODEL_PATH = "/lustre/project/ki-topml/minbui/projects/models/models--llava-hf--llava-v1.6-34b-hf/snapshots/66b6feb83d0249dc9f31a24bd3abfb63f90e41aa"
# MODEL_PATH = "/lustre/project/ki-topml/minbui/projects/models/models--llava-hf--llava-v1.6-vicuna-13b-hf/snapshots/e66fcaaa7d502b1037c8465375bb67f4c33758dd"
# MODEL_PATH = "/lustre/project/ki-topml/minbui/projects/models/sync/models--llava-hf--llava-next-72b-hf/snapshots/834da453803d866c3b45f4f94dc20f5b705d5a88"


def input_creator(all_prompts, image_paths, model_path, df_captions, add_caption):
    # Input for model_inference()
    processor = LlavaNextProcessor.from_pretrained(model_path)

    processor.patch_size = 14
    processor.vision_feature_select_strategy = "default"

    processed_prompts = []
    for image_path in image_paths:
        for raw_prompt in all_prompts:
            if add_caption:
                id_image = image_path.split("/")[-1].split(".jpg")[0]
                caption = df_captions[df_captions["ID"]
                                      == id_image]["Translation"].iloc[0]
                text_prompt = {"type": "text", "text": raw_prompt.format(str(caption))}
            else:
                text_prompt = {"type": "text", "text": raw_prompt}
            
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    text_prompt,
                ],
            },
            ]
            processed_prompt = processor.apply_chat_template(
                conversation, add_generation_prompt=True)
            processed_prompts.append(
                {"prompt": processed_prompt, "image_path": image_path})

    return processor, processed_prompts


def model_creator(model_path):
    model = LlavaNextForConditionalGeneration.from_pretrained(model_path,
                                                              torch_dtype=torch.float16,
                                                              low_cpu_mem_usage=True,
                                                              device_map="auto")
    return model


def model_inference(image_path, prompt, model, processor):
    raw_image = Image.open(image_path)
    inputs = processor(images=raw_image,
                       text=prompt,
                       return_tensors='pt')#.to(0, torch.float16)
    output = model.generate(**inputs, max_new_tokens=10,
                            do_sample=False, temperature=1.0)
    response_text = processor.decode(output[0][2:], skip_special_tokens=True)
    return response_text


if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Run pipeline inference with specified model path.')

    # Add an argument for MODEL_PATH
    parser.add_argument('--model_path', type=str, required=False, default='/lustre/project/ki-topml/minbui/projects/models/models--llava-hf--llava-v1.6-34b-hf/snapshots/66b6feb83d0249dc9f31a24bd3abfb63f90e41aa')
    parser.add_argument('--caption', action='store_true', help='Enable captioning')
    args = parser.parse_args()

    pipeline_inference(args.model_path, LANGUAGES, input_creator, model_creator, model_inference, add_caption=args.caption)
