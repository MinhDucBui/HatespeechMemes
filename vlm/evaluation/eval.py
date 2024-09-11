import pandas as pd
import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
two_dirs_up = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
sys.path.append(two_dirs_up)

from annotation_evaluation.utils import process_language_data
import argparse
from tqdm import tqdm

tqdm.pandas()

LANGUAGES = ["en", "de", "es", "hi", "zh"]
MAPPING = {
    "en": "US",
    "de": "DE",
    "es": "MX",
    "hi": "IN",
    "zh": "CN"
}

def extract_answer(response):
    response = response.lower()
    response = response.split("assistant: ")[-1]
    response = response.split("assistant\n")[-1]
    response = response.split(" ")[0].split("\n")[0].strip()
    return response 

def mapping_response(response):
    response = extract_answer(response)

    a_response = ["a", "a:", "a: hate", "a.", "a:"]
    b_response = ["b", "b:", "b:", "b)", "b:", "b."]

    invalid_response = ["bustin", "busters.kerry", "a.kerry", 
                        "busters.", "bhindiwereview", "bhaving",
                        "bheye."]
    if response in a_response:
        return 0
    elif response in b_response:
        return 1
    elif response in invalid_response:
        return -1
    else:
        raise ValueError(f"Invalid response encountered: {response}")


def process_response_to_hatespeech(row):
    # Invalid:
    if row['processed_answer'] == -1:
        return -1

    # if response = 0 -> a. if response = 1 -> b.
    if row['prompt'] % 2 != 0:
        # b = hate
        if row['processed_answer'] == 1:
            return 1
        elif row['processed_answer'] == 0:
            return 0
    else:
        # a = hate
        if row['processed_answer'] == 1:
            return 0
        elif row['processed_answer'] == 0:
            return 1


def calc_acc(df, gt_name, predict_name):
    correct_predictions = (df[gt_name] == df[predict_name]).sum()
    total_predictions = len(df)
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--annotation', '-a', type=str, default='/lustre/project/ki-topml/minbui/projects/MultiModalHatespeech/prolific/hatespeech_main')
    parser.add_argument('--model_inference', '-m', type=str, default='/lustre/project/ki-topml/minbui/projects/MultiModalHatespeech/model_predictions')
    args = parser.parse_args()
    df_gt = process_language_data(args.annotation)
    df_gt = df_gt.reset_index()
    df_gt["ID"] = df_gt["ID"].astype(str)

    # Loop over all folders inside the parent folder
    for root, dirs, files in os.walk(args.model_inference):
        for folder in dirs:
            if "models--" in folder:
                print("\n" + folder)
                for language in LANGUAGES:
                    print("Language: {}".format(language))
                    folder_path = os.path.join(root, folder, "responses_" + language + ".csv")
                    if os.path.exists(folder_path):
                        df_inference = pd.read_csv(folder_path)
                    else:
                        continue
                    df_inference['answer'] = df_inference['response'].apply(extract_answer)
                    df_inference['processed_answer'] = df_inference['response'].apply(mapping_response)
                    df_inference['hate_prediction'] = df_inference.apply(process_response_to_hatespeech, axis=1)
                    df_inference['response'] = df_inference["response"].str.replace("\n", "").str.replace("assistant", "").str[-20:]
                    df_inference = df_inference[["ID", "prompt", "response", "answer", "hate_prediction"]]
                    output_path = os.path.join(root, folder, "processed_responses_" + language + ".csv")
                    df_inference.to_csv(output_path, index=False)

                    # Evaluation 
                    df_inference["ID"] = df_inference["ID"].astype(str)
                    df_inference = pd.merge(df_inference, df_gt, on="ID")

                    # Accuracy
                    calc_acc(df_inference, MAPPING[language], "hate_prediction")

                    # N Invalid Responses
                    n_invalid = sum(df_inference["hate_prediction"] == -1)
                    print("Invalid responses: {}".format(n_invalid))
