import argparse
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
two_dirs_up = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
sys.path.append(two_dirs_up)

from vlm.inference.utils import pipeline_inference


MODEL_PATH = "/lustre/project/ki-topml/minbui/projects/models/models--llava-hf--llava-v1.6-vicuna-7b-hf/snapshots/89b0f2ea28da2e62d7cfda173a400d2ad82a1c8e"


def input_creator(all_prompts, image_paths, model_path):
    # Input for model_inference()
    processor = LlavaNextProcessor.from_pretrained(model_path)
    processed_prompts = []
    for image_path in image_paths:
        for raw_prompt in all_prompts:
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": raw_prompt},
                ],
            },
            ]
            processed_prompt = processor.apply_chat_template(
                conversation, add_generation_prompt=True)
            processed_prompts.append(
                {"prompt": processed_prompt, image_path: image_path})

    return processor, processed_prompts


def model_creator(model_path):
    model = LlavaNextForConditionalGeneration.from_pretrained(model_path,
                                                              torch_dtype=torch.float16,
                                                              low_cpu_mem_usage=True,
                                                              device_map="auto")
    return model


def model_inference(image_path, prompt, model, processor):
    raw_image = Image.open(image_path)
    inputs = processor(images=raw_image, text=prompt,
                       return_tensors='pt').to(0, torch.float16)
    output = model.generate(**inputs, max_new_tokens=10,
                            do_sample=False, temperature=1.0)
    response_text = processor.decode(output[0][2:], skip_special_tokens=True)
    return response_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--language', '-l', type=str, default='en')
    parser.add_argument('--annotation', '-a', type=str,
                        default='/lustre/project/ki-topml/minbui/projects/MultiModalHatespeech/prolific/hatespeech_main/')
    parser.add_argument('--output_folder', '-o', type=str,
                        default='/lustre/project/ki-topml/minbui/projects/MultiModalHatespeech/model_predictions')
    args = parser.parse_args()

    pipeline_inference(MODEL_PATH, args.language, args.annotation_path,
                       args.output_folder, input_creator, model_creator, model_inference)
