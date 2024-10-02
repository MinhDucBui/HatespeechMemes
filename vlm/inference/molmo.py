from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
from qwen_vl_utils import process_vision_info
import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
two_dirs_up = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
sys.path.append(two_dirs_up)
from vlm.inference.utils import pipeline_inference, create_prompt_for_input
import argparse


LANGUAGES = ["en", "de", "es", "hi", "zh"]
UNIMODAL = False

def input_creator(all_prompts, image_paths, model_path, df_captions, add_caption):
    # Input for model_inference()
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )
    processed_prompts = []
    for image_path in image_paths:
        for raw_prompt in all_prompts:
            text_prompt_1, text_prompt_2 = create_prompt_for_input(raw_prompt, df_captions, image_path, add_caption)

            if UNIMODAL:
                text_prompt_1["text"] = text_prompt_1["text"][:-7]
                text_prompt_2["text"] = text_prompt_2["text"].replace("Caption inside the meme:", "Text:")
                text_prompt_1["text"] = text_prompt_1["text"].replace("meme", "text")
                text_prompt_2["text"] = text_prompt_2["text"].replace("meme", "text")
                conversation = [{
                    "role": "user",
                    "content": [
                        text_prompt_1,
                        text_prompt_2,
                    ],
                },
                ]
            else:
                inputs = processor.process(
                            images=[Image.open(image_path)],
                            text=["Describe this image."]
                         )
                print(inputs)
                das
                conversation = [{
                    "role": "user",
                    "content": [
                        text_prompt_1,
                        {"type": "image"},
                        text_prompt_2
                    ],
                },
                ]

            processed_prompt = processor.apply_chat_template(
                conversation, add_generation_prompt=True)
            processed_prompts.append(
                {"prompt": [conversation, processed_prompt]})

    return processor, processed_prompts


def model_creator(model_path):
    # load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )
    return model


def model_inference(prompt, model, processor):
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in prompt.items()}
    output = model.generate_from_batch(
        prompt,
        GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer
    )

    # only get generated tokens; decode them to text
    generated_tokens = output[0,inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # print the generated text
    print(generated_text)
    return generated_text


if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Run pipeline inference with specified model path.')

    # Add an argument for MODEL_PATH
    parser.add_argument('--model_path', type=str, required=False, default='/lustre/project/ki-topml/minbui/projects/models/sync/models--allenai--Molmo-7B-D-0924/snapshots/b72f6745657cddaf97041d88eb02b23756338219')
    parser.add_argument('--caption', action='store_true', help='Enable captioning')
    args = parser.parse_args()

    pipeline_inference(args.model_path, LANGUAGES, input_creator, model_creator, model_inference, add_caption=args.caption)
