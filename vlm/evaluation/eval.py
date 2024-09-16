import pandas as pd
import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
two_dirs_up = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
sys.path.append(two_dirs_up)

from vlm.inference.local_paths import ANNOTATION_PATH, MODEL_PREDICTIONS
from annotation_evaluation.utils import process_language_data
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

def latex_table(latex_preds):

    if "en" in latex_preds:
        english = latex_preds["en"]
    else:
        english = {"US": "--", "DE": "--", "MX": "--", "CN": "--", "IN": "--"}
    if "de" in latex_preds:
        german = latex_preds["de"]
    else:
        german = {"US": "--", "DE": "--", "MX": "--", "CN": "--", "IN": "--"}
    if "es" in latex_preds:
        spanish = latex_preds["es"]
    else:
        spanish = {"US": "--", "DE": "--", "MX": "--", "CN": "--", "IN": "--"}
    if "hi" in latex_preds:
        hindi = latex_preds["hi"]
    else:
        hindi = {"US": "--", "DE": "--", "MX": "--", "CN": "--", "IN": "--"}
    if "zh" in latex_preds:
        mandarin = latex_preds["zh"]
    else:
        mandarin = {"US": "--", "DE": "--", "MX": "--", "CN": "--", "IN": "--"}


    table = f"""
    English   & {english["US"]} & {english["DE"]} & {english["MX"]} & {english["IN"]} & {english["CN"]} \\\\
    German    & {german["US"]} & {german["DE"]} & {german["MX"]} & {german["IN"]} & {german["CN"]} \\\\
    Spanish   & {spanish["US"]} & {spanish["DE"]} & {spanish["MX"]} & {spanish["IN"]} & {spanish["CN"]} \\\\
    Hindi     & {hindi["US"]} & {hindi["DE"]} & {hindi["MX"]} & {hindi["IN"]} & {hindi["CN"]} \\\\
    Mandarin  & {mandarin["US"]} & {mandarin["DE"]} & {mandarin["MX"]} & {mandarin["IN"]} & {mandarin["CN"]} \\\\
    """

    print(table)
    

def extract_answer(response):
    response = response.lower()
    response = response.split("assistant: ")[-1]
    response = response.split("assistant\n")[-1]
    response = response.split(" ")[0].split("\n")[0].strip()
    return response 

def mapping_response(response):
    response = extract_answer(response)

    a_response = ["a", "a:", "a: hate", "a.", "a:"]
    b_response = ["b", "b:", "b:", "b)", "b:", "b.", "a)"]

    invalid_response = ["bustin", "busters.kerry", "a.kerry", 
                        "busters.", "bhindiwereview", "bhaving",
                        "bheye.", "africaarabi.", "bilder.", "banner",
                        "banner", "bhaving", "bust", "bubble",
                        "b.19999999", "bietehrt.", "a120.", "ademon", "b20.",
                        "the"]
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
    # df = df[df[predict_name] != -1]
    correct_predictions = (df[gt_name] == df[predict_name]).sum()
    total_predictions = len(df)
    accuracy = (correct_predictions / total_predictions) * 100
    # print(f"Accuracy for GT Country {gt_name}: {accuracy:.2f}%")
    return accuracy


if __name__ == '__main__':
    
    df_gt = process_language_data(ANNOTATION_PATH)
    df_gt = df_gt.reset_index()
    df_gt["ID"] = df_gt["ID"].astype(str)

    # Loop over all folders inside the parent folder
    for root, dirs, files in os.walk(MODEL_PREDICTIONS):
        for folder in dirs:
            if "models--" in folder:
                print("\n--------------------" + folder + "-------------")
                latex_preds = {}
                for language in LANGUAGES:
                    # print("Language: {}".format(language))
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

                    latex_preds[language] = {}
                    # Accuracy
                    for language_eval in LANGUAGES:
                        accuracy = calc_acc(df_inference, MAPPING[language_eval], "hate_prediction")
                        latex_preds[language][MAPPING[language_eval]] = round(accuracy, 1)

                    # N Invalid Responses
                    n_invalid = sum(df_inference["hate_prediction"] == -1)
                    print("Invalid responses: {}".format(n_invalid))

                latex_table(latex_preds)
