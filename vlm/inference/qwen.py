from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
two_dirs_up = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
sys.path.append(two_dirs_up)
from vlm.inference.utils import pipeline_inference


LANGUAGES = ["en", "de", "es", "hi", "zh"]
MODEL_PATH = "/lustre/project/ki-topml/minbui/projects/models/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/3ca981c995b0ce691d85d8408216da11ff92f690"


def input_creator(all_prompts, image_paths, model_path, df_captions, add_caption):
    # Input for model_inference()
    processor = AutoProcessor.from_pretrained(model_path)
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
                    {"type": "image", "image": image_path},
                    text_prompt,
                ],
            },
            ]
            processed_prompt = processor.apply_chat_template(
                conversation, add_generation_prompt=True)
            processed_prompts.append(
                {"prompt": [conversation, processed_prompt]})

    return processor, processed_prompts


def model_creator(model_path):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    return model


def model_inference(prompt, model, processor):
    image_inputs, video_inputs = process_vision_info(prompt[0])
    inputs = processor(
        text=[prompt[1]],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    response_text = processor.decode(output[0][2:], skip_special_tokens=True)
    return response_text


if __name__ == '__main__':
    for language in LANGUAGES:
        pipeline_inference(MODEL_PATH, language, input_creator, model_creator, model_inference)
