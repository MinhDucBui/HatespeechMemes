from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import argparse
import os
import tiktoken
import json


API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)
MODEL_NAME = "gpt-3.5-turbo-0125"
LANGUAGE = "en"

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


def tokenize(prompt, tokenizer, return_tensors=None,
             cutoff_len=500, padding=True, add_special_tokens=True):
    if padding:
        return tokenizer(
            prompt + tokenizer.eos_token,
            truncation=True,
            max_length=cutoff_len,
            padding="max_length",
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens
        )
    else:
        return tokenizer(
            prompt + tokenizer.eos_token,
            truncation=True,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens
        )


def calculate_token_count(prompts, encoding):
    token_count = sum(
        len(encoding.encode(unit["content"])) for prompt in prompts for unit in prompt)
    return token_count


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


def save_into_jsonl(all_prompts, filtered_df, batch_size=800, prefix=""):
    # Define the number of prompts per batch
    prompts_per_batch = batch_size

    # Divide the prompts into batches
    prompt_batches = [all_prompts[i:i + prompts_per_batch]
                      for i in range(0, len(all_prompts), prompts_per_batch)]

    # Iterate over each batch and create a JSONL file
    for batch_index, batch in enumerate(prompt_batches):
        # Define the file name for the batch
        jsonl_file = f"{prefix}_batch_{batch_index}.jsonl"

        # Open the file in write mode
        with open(jsonl_file, "w") as f:
            # Iterate over each prompt in the batch and write to the JSONL file
            for index, prompt in enumerate(batch):
                custom_id = str(filtered_df[index]["instance_id"])
                json_row = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-3.5-turbo-0125",
                        "messages": prompt,
                        "max_tokens": 1000
                    }
                }
                # Write the JSON object to the file with a newline character
                f.write(json.dumps(json_row) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--captions_path', '-c', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/generated_memes/caption_translation')
    parser.add_argument('--save_folder', '-s', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/generated_memes')
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

    # Calculate Costs
    encoding = tiktoken.encoding_for_model(MODEL_NAME)
    token_count = calculate_token_count(all_prompts, encoding)
    print("Token count:", token_count)
    print("Average tokens per prompt:", token_count / len(all_prompts))
    print("Total Cost (GPT-4):", token_count / 1_000_000 * 10)
    print("Total Cost (GPT-3.5):", token_count / 1_000_000 * 0.5)

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

    for index, prompt in tqdm(enumerate(all_prompts), total=len(all_prompts)):
        if existing_ids and filtered_df.iloc[index]["instance_id"] in existing_ids:
            continue
        # print(filtered_df.iloc[index]["template"])
        if index % saving_iter == 0 and index != 0:
            subset_df = filtered_df[:index].copy()
            save_results(subset_df, save_answers, save_folder)

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=prompt,
                temperature=0,
            )

            # save_prompt.append(current_prompt_str)  # Uncomment if needed
            save_answers.append(response.choices[0].message.content)
        except Exception as e:
            print(f"Error occurred: {e}")
            save_answers.append("")
    subset_df = filtered_df.copy()
    save_results(subset_df, save_answers, save_folder)
