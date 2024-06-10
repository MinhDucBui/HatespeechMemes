from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import argparse
import os
import tiktoken
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


LANGUAGE = "en"
#MODEL_NAME = "models--meta-llama--Meta-Llama-3-70B-Instruct"
MODEL_NAME = "41b61a33a2483885c981aa79e0df6b32407ed873"
MODEL_FOLDER_PATH = "/lustre/project/ki-topml/minbui/projects/models/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots"

MODEL_FOLDER_PATH = "/lustre/project/ki-topml/mcascino/models"
MODEL_NAME = "models--meta-llama--Meta-Llama-3-70B-Instruct"

HATESPEECH_DEFINITION = "A direct or indirect attack on people based on characteristics, including ethnicity, race, nationality, immigration status, religion, caste, sex, gender identity, sexual orientation, and disability or disease. We define attack as violent or dehumanizing (comparing people to non-human things, e.g. animals) speech, statements of inferiority, and calls for exclusion or segregation. Mocking hate crime is also considered hate speech."


def prompt_prefix(multimodal_setting):
    message = [
        {"role": "system", "content": "You are a helpful assistant who follows the following pattern."},
    ]

    if multimodal_setting:
        message = message + [{"role": "user",
                              "content": f"Classify the following meme into 'hatespeech' or 'no hatespeech'. "
                              f"The definition of hatespeech is: {HATESPEECH_DEFINITION}\n"
                              "Reason and then Answer with 'Yes' or 'No'. Format is 'Reason: <Your Reasoning>\nAnswer: <Yes or No>'.\n"}
                             ]
    else:
        message = message + [{"role": "user",
                              "content": f"Classify the following text into 'hatespeech' or 'no hatespeech'. "
                              f"The definition of hatespeech is: {HATESPEECH_DEFINITION}\n"
                              "Reason and then Answer with 'Yes' or 'No'. Format is 'Reason: <Your Reasoning>\nAnswer: <Yes or No>'.\n"}
                             ]

    return message


def image_description(template):
    template = template.replace("-", " ").lower().strip()
    if template == "scumbag america":
        description = "Meme Image: Depicts america, with a hat on from the meme 'scumbag steve'."

    elif template == "advice hitler":
        description = "Meme Image: Depicts hitler."

    elif template == "angry black woman":
        description = "Meme Image: Depicts a black woman."

    elif template == "scumbag america":
        description = "Meme Image: Depicts america, with a hat on from the meme 'scumbag steve'."

    elif template == "scumbag america":
        description = "Meme Image: Depicts america, with a hat on from the meme 'scumbag steve'."

    elif template == "scumbag america":
        description = "Meme Image: Depicts america, with a hat on from the meme 'scumbag steve'."

    elif template == "scumbag catholic priest":
        description = "Meme Image: Depicts a catholic priest."

    elif template == "confused muslim girl":
        description = "Meme Image: Depicts a confused muslim girl with a headscarf."

    elif template == "angry muslim guy":
        description = "Meme Image: Depicts an angry muslim."

    elif template == "advicejew":
        description = "Meme Image: Depicts a jew with israelin flag."

    elif template == "ordinary muslim man":
        description = "Meme Image: Depicts a muslim."

    elif template == "stereotypical indian telemarketer":
        description = "Meme Image: A female indian wearing a headset."

    elif template == "generic indian guy":
        description = "Meme Image: Depicts a male indian."

    elif template == "african children dancing":
        description = "Meme Image: Depicts young black childen dancing with little clothing."

    elif template == "american pride eagle":
        description = "Meme Image: Depicts an eagle with american flag in the background."

    elif template == "successful mexican":
        description = "Meme Image: Depicts a mexican with a big sombrero."

    elif template == "provincial man":
        description = "Meme Image: Depicts a bald (provincial) man with thick, close-set eyebrows, a broad smile, and a mustache."

    elif template == "advice hitler":
        description = "Meme Image: Depicts hitler."

    elif template == "feministfrequently":
        description = "Meme Image: Depicts an angry woman."

    elif template == "feminist cunt":
        description = "Meme Image: Depicts a woman with a tshirt referencing feminism."

    elif template == "liberal douche garofalo":
        description = "Meme Image: Depicts Garofalo, who is a liberal."

    elif template == "pms woman":
        description = "Meme Image: Depicts a 50s cartoon woman."

    elif template == "skinny kid":
        description = "Meme Image: Depicts a skinny looking asian kid."

    elif template == "fat girl":
        description = "Meme Image: Depicts an overweight girl."

    elif template == "fat girl":
        description = "Meme Image: Depicts an overweight girl."

    elif template == "obama":
        description = "Meme Image: Depicts obama."

    elif template == "obama laughing":
        description = "Meme Image: Depicts obam laughing."

    elif template == "laughing girls":
        description = "Meme Image: Depicts two white blonde females laughing."

    elif template == "asian college freshman":
        description = "Meme Image: Depicts an asian in college."

    elif template == "scumbag whitehouse":
        description = "Meme Image: Depicts the whitehouse with a cap on."

    elif template == "stereotypical redneck":
        description = "Meme Image: Depicts a white person leaning on a truck (supposed to depict a 'redneck')."

    elif template == "canada flag":
        description = "Meme Image: Depicts the canadian flag."

    elif template == "canada flag":
        description = "Meme Image: Depicts the canadian flag."

    else:
        raise ValueError(f"No Image Description for {template}.")
    return description


def tokenize(prompts, tokenizer, return_tensors=None,
             cutoff_len=256, padding=True, add_special_tokens=True):
    prompts = [prompt + tokenizer.eos_token for prompt in prompts]
    if padding:
        return tokenizer(
            prompts,
            truncation=True,
            max_length=cutoff_len,
            padding="max_length",
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens
        )
    else:
        return tokenizer(
            prompts,
            truncation=True,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens
        )



def load_model(model_name, base_path = '/lustre/project/ki-topml/mcascino/models/', cache_path = None):

    model_name = model_name.replace("/", "--")
    model_path = os.path.join(base_path, model_name)
    if cache_path is None:
        cache_dir = model_path
    else:
        cache_dir = cache_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir = cache_dir, device_map = 'auto', padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir = cache_dir, device_map = 'auto', local_files_only = True)
    #+model = None
    return model, tokenizer


def save_results(df, answers, save_folder):
    df["prediction"] = answers
    folder_path = os.path.join(save_folder, "multimodal_detection")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if multimodal_setting:
        df.to_csv(os.path.join(folder_path, 'multimodal_detection.csv'),
                  index=False)
    else:
        df.to_csv(os.path.join(folder_path, 'text_detection.csv'),
                  index=False)

def inference(model, tokenizer, prompt):
    output = model.generate(**prompt, max_new_tokens=300)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


def batch_inference(model, tokenizer, prompts):
    """
    Perform inference on a batch of prompts.
    
    Parameters:
    - model: The language model used for generating text.
    - tokenizer: The tokenizer used to encode and decode text.
    - prompts: A list of strings, where each string is a prompt for the model.
    
    Returns:
    - List of generated texts for each prompt.
    """
    # Tokenize all prompts in the batch
    #batch_input = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
    
    # Generate outputs for the batch
    with torch.no_grad():
        outputs = model.generate(**prompts, max_new_tokens=300)
    
    # Decode each output in the batch
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    return generated_texts

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--captions_path', '-c', type=str,
                        default='/lustre/project/ki-topml/minbui/projects/MultiModalHatespeech/dataset/generated_memes/caption_translation')
    parser.add_argument('--save_folder', '-s', type=str,
                        default='/lustre/project/ki-topml/minbui/projects/MultiModalHatespeech/dataset/generated_memes')
    parser.add_argument('--multimodal', '-m',
                        action='store_true', default=False)

    args = parser.parse_args()

    captions_path = args.captions_path
    save_folder = args.save_folder
    multimodal_setting = args.multimodal
    captions_path = os.path.join(captions_path, LANGUAGE + ".txt")

    # Assuming the file name is 'your_text_file.txt'
    # Replace with your actual column names
    headers = ['template', 'instance_id', 'caption', 'original']

    # Load the text file into a DataFrame
    df = pd.read_csv(captions_path, sep='\t', names=headers)
    df = df.iloc[:1000]
    # df = df.iloc[-30:]
    all_prompts = []
    filtered_df = []  # Create empty dataframe

    for index_dev, (_, row) in enumerate(df.iterrows()):
        template = row["template"].split("_")[0].replace("-", " ").lower()
        if template == "transvestite trevor" \
                or template == "scumbag god" \
                or template == "gay richard simmons" \
                or template == "sassy black woman" \
                or template == "chinese lesbians" \
                or template == "homeless man 2" \
                or template == "asinine america" \
                or template == "rich men laughing":
            continue
        message = prompt_prefix(multimodal_setting)
        text = row["caption"]
        text = text.replace("<sep>", "-")
        if multimodal_setting:
            description = image_description(template)
            message[-1]["content"] = message[-1]["content"] + description

        message[-1]["content"] = message[-1]["content"] + " Text: " + text
        all_prompts.append(message)
        filtered_df.append(row)


    # 
    model, tokenizer = load_model(MODEL_NAME, base_path=MODEL_FOLDER_PATH,
                                  cache_path="/lustre/project/ki-topml/minbui/projects/models/cache")
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_prompts = []
    prompts_template = tokenizer.apply_chat_template(all_prompts, tokenize=False, add_generation_prompt=True)
    tokenized = tokenize(prompts_template, tokenizer, return_tensors="pt")
    tokenized = {key: value.to('cuda') for key, value in tokenized.items()}

    # Check if there is any row with only 1s
    # row_all_ones = torch.all(tokenized["attention_mask"] == 1, dim=1)
    # Find out if there's any row with only 1s
    # any_row_all_ones = torch.any(row_all_ones)
    # print(f"Is there any row with only 1s? {any_row_all_ones}")

    #for prompt in tqdm(all_prompts):
    #    prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    #    tokenized = tokenize(prompt, tokenizer, return_tensors="pt")
    #    tokenized = {key: value.to('cuda') for key, value in tokenized.items()}
    #    tokenized_prompts.append(tokenized)

    save_answers = []
    saving_iter = 100

    # if multimodal_setting:
    #    save_into_jsonl(all_prompts, filtered_df, prefix="multimodal")
    # else:
    #    save_into_jsonl(all_prompts, filtered_df, prefix="text")

    filtered_df = pd.DataFrame(filtered_df)
    # Path
    path_existing = os.path.join(save_folder, "multimodal_detection")
    if multimodal_setting:
        path_existing = os.path.join(path_existing, 'multimodal_detection.csv')
    else:
        path_existing = os.path.join(path_existing, 'text_detection.csv')
    if os.path.isfile(path_existing):
        df_existing = pd.read_csv(path_existing)
        save_answers = list(df_existing["prediction"])
        existing_ids = list(df_existing["instance_id"])
    else:
        existing_ids = []

    batch_size = 16  # Adjust this number based on your memory constraints and requirements
    saving_iter = 16
    for i in tqdm(range(0, tokenized['input_ids'].size(0), batch_size)):
        batch = {key: value[i:i+batch_size] for key, value in tokenized.items()}
        batch_answers = batch_inference(model, tokenizer, batch)
        save_answers.extend(batch_answers)
        index = i+batch_size
        if index % saving_iter == 0 and index != 0:
            subset_df = filtered_df[:index].copy()
            save_results(subset_df, save_answers, save_folder)
    subset_df = filtered_df.copy()
    save_results(subset_df, save_answers, save_folder)
