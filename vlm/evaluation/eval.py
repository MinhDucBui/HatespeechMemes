import pandas as pd
import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
two_dirs_up = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
sys.path.append(two_dirs_up)

from vlm.inference.local_paths import ANNOTATION_PATH, MODEL_PREDICTIONS
from annotation_evaluation.utils import process_language_data
from tqdm import tqdm
import numpy as np
from scipy.stats import ranksums
from sklearn.metrics import f1_score


tqdm.pandas()

LANGUAGES = ["en", "de", "es", "hi", "zh"]
MAPPING = {
    "en": "US",
    "de": "DE",
    "es": "MX",
    "hi": "IN",
    "zh": "CN"
}

def bold_underline_value(list_values):
    list_copy = list_values.copy()
    for key, value in list_values.items():
        if "--" == value:
            return list_values

    max_key = max(list_values, key=lambda k: float(list_values[k]))
    min_key = min(list_values, key=lambda k: float(list_values[k]))

    max_value = str(list_values[max_key])
    bolding = "\\textbf{" + max_value + "}"
    list_copy[max_key] = bolding

    min_value = str(list_values[min_key])
    bolding = "\\underline{" + min_value + "}"
    list_copy[min_key] = bolding

    return list_copy

def bold_underline_value_caption(list_values1, list_values2):
    list1_copy = list_values1.copy()
    list2_copy = list_values2.copy()
    for key, value in list_values1.items():
        if "--" == value or list2_copy[key] == "--":
            return list_values1

    for key in list1_copy.keys():
        if list1_copy[key] > list2_copy[key]:
            list1_copy[key] = "\\textbf{" + str(list1_copy[key]) + "}"
        else:
            list2_copy[key] = "\\textbf{" + str(list2_copy[key]) + "}"

    return list1_copy, list2_copy


def latex_table(latex_preds):

    for model, preds in latex_preds.items():
        #if "_caption" in model:
        #    continue
        print("----------{}-------".format(model))
        default_vals = {"US": "--", "DE": "--", "MX": "--", "CN": "--", "IN": "--"}
        languages = ["en", "de", "es", "hi", "zh"]
        compare_two = False
        if compare_two:
            if model + "_caption" in latex_preds:
                preds_capt = latex_preds[model + "_caption"]
                [english, english_capt], [german, german_capt], [spanish, spanish_capt], [hindi, hindi_capt], [mandarin, mandarin_capt] = (
                    bold_underline_value_caption(preds.get(lang, default_vals), preds_capt.get(lang, default_vals)) for lang in languages
                )
            else:
                continue

        else:
            english, german, spanish, hindi, mandarin = (
                bold_underline_value(preds.get(lang, default_vals)) for lang in languages
            )

            if model + "_caption" in latex_preds:
                preds_capt = latex_preds[model + "_caption"]
                english_capt, german_capt, spanish_capt, hindi_capt, mandarin_capt = (
                    bold_underline_value(preds_capt.get(lang, default_vals)) for lang in languages
                )
            else:
                english_capt, german_capt, spanish_capt, hindi_capt, mandarin_capt = (
                    bold_underline_value(default_vals) for lang in languages
                )


        table = f"""
        English   & {english["US"]} & {english["DE"]} & {english["MX"]} & {english["IN"]} & {english["CN"]} \\\\ \\hline
        \quad + Caption   & {english_capt["US"]} & {english_capt["DE"]} & {english_capt["MX"]} & {english_capt["IN"]} & {english_capt["CN"]} \\\\ \\hline
        German    & {german["US"]} & {german["DE"]} & {german["MX"]} & {german["IN"]} & {german["CN"]} \\\\
        \quad + Caption   & {german_capt["US"]} & {german_capt["DE"]} & {german_capt["MX"]} & {german_capt["IN"]} & {german_capt["CN"]} \\\\ \\hline
        Spanish   & {spanish["US"]} & {spanish["DE"]} & {spanish["MX"]} & {spanish["IN"]} & {spanish["CN"]} \\\\
        \quad + Caption   & {spanish_capt["US"]} & {spanish_capt["DE"]} & {spanish_capt["MX"]} & {spanish_capt["IN"]} & {spanish_capt["CN"]} \\\\ \\hline
        Hindi     & {hindi["US"]} & {hindi["DE"]} & {hindi["MX"]} & {hindi["IN"]} & {hindi["CN"]} \\\\
        \quad + Caption   & {hindi_capt["US"]} & {hindi_capt["DE"]} & {hindi_capt["MX"]} & {hindi_capt["IN"]} & {hindi_capt["CN"]} \\\\
        Mandarin  & {mandarin["US"]} & {mandarin["DE"]} & {mandarin["MX"]} & {mandarin["IN"]} & {mandarin["CN"]} \\\\
        \quad + Caption   & {mandarin_capt["US"]} & {mandarin_capt["DE"]} & {mandarin_capt["MX"]} & {mandarin_capt["IN"]} & {mandarin_capt["CN"]} \\\\
        """

        table = f"""
        English   & {english["US"]} & {english["DE"]} & {english["MX"]} & {english["IN"]} & {english["CN"]} \\\\
        \quad + Caption   & {english_capt["US"]} & {english_capt["DE"]} & {english_capt["MX"]} & {english_capt["IN"]} & {english_capt["CN"]} \\\\ \\hline
        German    & {german["US"]} & {german["DE"]} & {german["MX"]} & {german["IN"]} & {german["CN"]} \\\\
        \quad + Caption   & {german_capt["US"]} & {german_capt["DE"]} & {german_capt["MX"]} & {german_capt["IN"]} & {german_capt["CN"]} \\\\ \\hline
        Spanish   & {spanish["US"]} & {spanish["DE"]} & {spanish["MX"]} & {spanish["IN"]} & {spanish["CN"]} \\\\
        \quad + Caption   & {spanish_capt["US"]} & {spanish_capt["DE"]} & {spanish_capt["MX"]} & {spanish_capt["IN"]} & {spanish_capt["CN"]} \\\\ \\hline
        Hindi     & {hindi["US"]} & {hindi["DE"]} & {hindi["MX"]} & {hindi["IN"]} & {hindi["CN"]} \\\\
        \quad + Caption   & {hindi_capt["US"]} & {hindi_capt["DE"]} & {hindi_capt["MX"]} & {hindi_capt["IN"]} & {hindi_capt["CN"]} \\\\  \\hline
        Mandarin  & {mandarin["US"]} & {mandarin["DE"]} & {mandarin["MX"]} & {mandarin["IN"]} & {mandarin["CN"]} \\\\
        \quad + Caption   & {mandarin_capt["US"]} & {mandarin_capt["DE"]} & {mandarin_capt["MX"]} & {mandarin_capt["IN"]} & {mandarin_capt["CN"]} \\\\
        """


        table = f"""
        English   & {english_capt["US"]} & {english_capt["DE"]} & {english_capt["MX"]} & {english_capt["IN"]} & {english_capt["CN"]} \\\\
        German   & {german_capt["US"]} & {german_capt["DE"]} & {german_capt["MX"]} & {german_capt["IN"]} & {german_capt["CN"]} \\\\
        Spanish   & {spanish_capt["US"]} & {spanish_capt["DE"]} & {spanish_capt["MX"]} & {spanish_capt["IN"]} & {spanish_capt["CN"]} \\\\
        Hindi   & {hindi_capt["US"]} & {hindi_capt["DE"]} & {hindi_capt["MX"]} & {hindi_capt["IN"]} & {hindi_capt["CN"]} \\\\
        Mandarin   & {mandarin_capt["US"]} & {mandarin_capt["DE"]} & {mandarin_capt["MX"]} & {mandarin_capt["IN"]} & {mandarin_capt["CN"]} \\\\
        """
        print(table)
    

def extract_answer(response):
    response = response.lower()
    response = response.split("assistant: ")[-1]
    response = response.split("assistant:")[-1]
    response = response.split("assistant\n")[-1]
    response = response.split("answer choice:")[-1]
    response = response.split("answer:")[-1]
    response = response.split("**answer**:")[-1]
    response = response.strip()

    b_responses = ["b: hate", "b: non-hate", "answer: b"]
    a_responses = ["a: non-hate", "a: hate", "a) hate", "answer: a"]
    if any(s in response for s in b_responses):
        return "b"
    if any(s in response for s in a_responses):
        return "a"

    response = response.split(" ")[0].split("\n")[0].strip()
    return response 

def mapping_response(response):
    extracted = extract_answer(response)

    a_response = ["a", "a:", "a: hate", "a.", "a:", "a)", "a (**Non-hate**)", "'a'"]
    b_response = ["b", "b:", "b:", "b)", "b:", "b.", "b: hate", "b: non-hate"]

    invalid_response = ["bustin", "busters.kerry", "a.kerry", 
                        "busters.", "bhindiwereview", "bhaving",
                        "bheye.", "africaarabi.", "bilder.", "banner",
                        "banner", "bhaving", "bust", "bubble",
                        "b.19999999", "bietehrt.", "a120.", "ademon", "b20.",
                        "the"]

    # Some more logic:
    response_ = response.split("Assistant: ")[-1]
    if "This meme is non-hate" in response_ or "no." in extracted or "no" == extracted or "non-hate" == extracted or "non-hate." == extracted or "Not hate" == response_ or "Not hate." == response_ or "Not hate speech" == response_ or "Not hate speech." == response_:
        if "b: Non-hate" in response:
            return 1
        elif "a: Non-hate" in response:
            return 0
    if "hate" in extracted or "yes" in extracted or "This meme is hate speech" in response_ or "This meme can be considered Hate (a)" in response_:
        if "b: Hate" in response:
            return 1
        elif "a: Hate" in response:
            return 0
    if "it's not possible to confidently determine" in response or "I can't assist" in response or "I must respectfully decline" in response or "I will do my best to help you." in response or "No: Hate" in response or "I cannot view or acknowledge contents" in response or "I would not be able to determine" in response or "I'm unable to provide assistance" in response or "There isn’t enough context" in response or "I cannot confidently provide a definitive answer" in response or "I am unable to determine" in response or "I am unable to provide a detailed analysis" in response or "Please provide the image you'd like to evaluate" in response or "To determine whether this specific meme qualifies as hate speech, one would need to directly analyz" in response or "Without understanding the exact translation or context of the text in the meme" in response or "I'm unable to determine" in response or "I cannot provide a response" in response or "I am unable to determine if the meme is hateful or non-hateful since" in response or "I cannot provide detailed textual analysis of the content in images" in response or "As an AI visual assistant, I do not have the capability to translate" in response or "I cannot assist with tha" in response or "Since I do not understand the language of the text in the meme" in response or extracted in " c " or "Please provide a brief " in response or "Since I'm unable to understand the context of the meme without understanding" in response or "I'm sorry, I can't respond to that request" in response or "I cannot provide a response for this question" in response or "I wouldn't be able to accurately evaluate the meme as its content is not accessible to me" in response or "I cannot provide an analysis based on the text you've provided." in response or "Please provide the text or details from the meme for me" in response or "It is difficult to determine definitively without full context." in response or "Without more additional context, I am unable to assess whether this specific meme is hate speech or not." in response or "I am unable to provide a response as the content of the meme may contain harmful or biased language." in response or "I'm sorry, but this content exceeds the limits of appropriateness that I am designed to handle." in response or "I can't give an answer" in response:
        return -1

    if extracted in a_response or "a. Non-hate" in response or "option a" in response or "a. Hate" in response:
        return 0
    elif extracted in b_response:
        return 1
    elif extracted in invalid_response:
        return -1
    else:
        raise ValueError(f"Invalid response encountered: {response} \n Extracted: {extracted}")


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


# Function to calculate accuracy with dynamic column names
def calculate_group_accuracy(df, group_col, col1, col2):
    df_group = df.groupby(group_col)
    accuracy = df_group[[col1, col2]].apply(lambda x: (x[col1] == x[col2]).mean()).reset_index(name='accuracy')
    return accuracy

def stat_test(df, gt_name, predict_name):
    accuracy_by_group = calculate_group_accuracy(df, "ID", gt_name, predict_name)
    accuracy_by_group_us = calculate_group_accuracy(df, "ID", "US", predict_name)
    sample1 = np.array(accuracy_by_group_us["accuracy"])
    sample2 = np.array(accuracy_by_group["accuracy"])
    a = round(ranksums(sample1, sample2).pvalue, 5)
    # print("Statistic: for {} and {}: {}".format(gt_name, "US", a))


def calc_acc(df, gt_name, predict_name):
    # df = df[df[predict_name] != -1]

    # stat_test(df, gt_name, predict_name)
    # df = df[df["prompt"] == 0]
    correct_predictions = (df[gt_name] == df[predict_name]).sum()
    total_predictions = len(df)
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Accuracy for GT Country {gt_name}: {accuracy:.2f}%")

    # Calculate F1 score
    # y_true = list(df[gt_name])
    # y_pred = list(df[predict_name])
    # y_pred = [y_true[i] == 0 if pred == -1 else pred for i, pred in enumerate(y_pred)]
    # f1 = f1_score(y_true, y_pred, average='binary') * 100  # Use 'macro' for multi-class classification

    return accuracy


if __name__ == '__main__':
    
    df_gt = process_language_data(ANNOTATION_PATH)
    df_gt = df_gt.reset_index()
    df_gt["ID"] = df_gt["ID"].astype(str)

    latex_preds = {}
    # Loop over all folders inside the parent folder
    for root, dirs, files in os.walk(MODEL_PREDICTIONS):
        for folder in dirs:
            if "archive" in root:
                continue
            if "models--" in folder:
                latex_preds[folder] = {}
                print("\n--------------------" + folder + "-------------")
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

                    latex_preds[folder][language] = {}
                    # Accuracy
                    for language_eval in LANGUAGES:
                        accuracy = calc_acc(df_inference, MAPPING[language_eval], "hate_prediction")
                        latex_preds[folder][language][MAPPING[language_eval]] = round(accuracy, 1)

                    # N Invalid Responses
                    n_invalid = sum(df_inference["hate_prediction"] == -1)
                    print("Invalid responses: {}".format(n_invalid))

    latex_table(latex_preds)
