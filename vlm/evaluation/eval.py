import pandas as pd
import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
two_dirs_up = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
sys.path.append(two_dirs_up)
import copy
from vlm.inference.local_paths import ANNOTATION_PATH, MODEL_PREDICTIONS
from annotation_evaluation.utils import process_language_data
from tqdm import tqdm
import numpy as np
from scipy.stats import ranksums
from sklearn.metrics import f1_score
import numpy as np
from scipy import stats
import re
np.seterr(divide='ignore', invalid='ignore')


tqdm.pandas()

LANGUAGES = ["en", "de", "es", "hi", "zh"]
MAPPING = {
    "en": "US",
    "de": "DE",
    "es": "MX",
    "hi": "IN",
    "zh": "CN"
}
EVAL_MODELS = "new-"
TABLE_IDX = 0

def bold_underline_value(list_values):
    list_copy = copy.deepcopy(list_values)
    for key, value in list_values.items():
        if "--" == value:
            return list_values

    # Sort the keys based on the values and get the second largest
    sorted_keys = sorted(list_values, key=lambda k: float(list_values[k][0]), reverse=True)
    second_max_key = sorted_keys[1] if len(sorted_keys) > 1 else None
    max_key = max(list_values, key=lambda k: float(list_values[k][0]))
    min_key = min(list_values, key=lambda k: float(list_values[k][0]))

    max_value = str(list_values[max_key][0])
    bolding = "\\textbf{" + max_value + "}"
    list_copy[max_key][0] = bolding

    min_value = str(list_values[min_key][0])
    bolding = "\\underline{" + min_value + "}"
    list_copy[min_key][0] = bolding

    min_value = str(list_values[second_max_key][0])
    bolding = "\\phantom{}" + min_value
    list_copy[second_max_key][0] = bolding
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



def significance_test(row):
    n1 = 6
    n2 = 6
    min_max_value = [value for key, value in row.items() if "textbf" in str(value[0]) or "underline" in str(value[0]) or "phantom" in str(value[0])]
    for index, value in enumerate(min_max_value):
        match = re.search(r'\\(?:textbf|underline){([\d.]*)}', value[0])
        if match:
            min_max_value[index] = [float(match.group(1)), value[1], value[3]]
        if "phantom" in value[0]:
            value[0] = value[0].replace("\\phantom{}", "")
            min_max_value[index] = [float(value[0]), value[1], value[3]]

    min_max_value = sorted(min_max_value, key=lambda x: x[0])
    if min_max_value and len(min_max_value) == 3:
        # T Test
        #_, p_value = stats.ttest_ind_from_stats(min_max_value[0][0], min_max_value[0][1], n1, min_max_value[1][0], min_max_value[1][1], n2, equal_var=False)
        
        # Wilcoxon Rank Sum
        p_value = stat_test(min_max_value[0][2], min_max_value[2][2])
        alpha = 0.05
        if p_value < alpha:
            for key, value in row.items():
                if "textbf" in str(value[0]):
                    p_value = stat_test(min_max_value[0][2], min_max_value[1][2])
                    if p_value < alpha:
                        row[key] = [value[0] + "$^{\mathbf{**}}", value[1]]
                    else:
                        row[key] = [value[0] + "$^{\mathbf{*}}", value[1]]

            return row
        else:
            return row
    else:
        return row


# def add_std_to_latex():


def latex_table(latex_preds):
    def format_value(value, key="MX"):
        """Helper function to format values as LaTeX strings."""
        if value == "--":
            return ""
        if "*" in str(value[0]):
            text = f"{value[0]}_{{\pm{value[1]}}}$"
        else:
            text = f"{value[0]}$_{{\pm {value[1]}}}$"

        #if key == "US":
        #    text = " {} & ".format(value[2]) + text
        return text

    def process_language(preds, languages, default_vals, bold_underline_func):
        """Processes predictions for a given set of languages."""
        return {
            lang: significance_test(bold_underline_func(preds.get(lang, default_vals))) for lang in languages
        }

    def format_language_results(lang_dict):
        """Formats the language dictionary into LaTeX-compliant strings."""
        return {key: format_value(value, key) for key, value in lang_dict.items()}

    # Sample DataFrame for defaults
    default_vals = {"US": "--", "DE": "--", "MX": "--", "CN": "--", "IN": "--"}
    languages = ["en", "de", "es", "hi", "zh"]

    # Iterate through models and predictions
    for model, preds in latex_preds.items():
        print(f"----------{model}-------")
        
        # Determine whether to compare two sets of predictions
        compare_two = False
        if compare_two and model + "_caption" in latex_preds:
            preds_capt = latex_preds[model + "_caption"]
            language_stats = [
                significance_test(bold_underline_value_caption(preds.get(lang, default_vals)), 
                                significance_test(preds_capt.get(lang, default_vals)))
                for lang in languages
            ]
        else:
            # Process predictions for the main and caption model (if available)
            language_stats = process_language(preds, languages, default_vals, bold_underline_value)
            if model + "_caption" in latex_preds:
                preds_capt = latex_preds[model + "_caption"]
                caption_stats = process_language(preds_capt, languages, default_vals, bold_underline_value)
            else:
                caption_stats = process_language(default_vals, languages, default_vals, bold_underline_value)

        # Format the results for LaTeX
        for lang, value in language_stats.items():
            for country in value.keys():
                if language_stats[lang][country] != "--":
                    language_stats[lang][country].append(preds[lang][country][2])
                if caption_stats and caption_stats[lang][country] != "--":
                    caption_stats[lang][country].append(preds_capt[lang][country][2])

        english, german, spanish, hindi, mandarin = map(format_language_results, language_stats.values())
        english_capt, german_capt, spanish_capt, hindi_capt, mandarin_capt = map(format_language_results, caption_stats.values())

        
        if TABLE_IDX == 0:
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
        elif TABLE_IDX == 1:
            table = f"""
            English   &  {english_capt["US"]} & {english_capt["DE"]} & {english_capt["MX"]} & {english_capt["IN"]} & {english_capt["CN"]} \\\\ 
            German   & {german_capt["US"]} & {german_capt["DE"]} & {german_capt["MX"]} & {german_capt["IN"]} & {german_capt["CN"]} \\\\ 
            Spanish   &  {spanish_capt["US"]} & {spanish_capt["DE"]} & {spanish_capt["MX"]} & {spanish_capt["IN"]} & {spanish_capt["CN"]} \\\\ 
            Hindi     & {hindi_capt["US"]} & {hindi_capt["DE"]} & {hindi_capt["MX"]} & {hindi_capt["IN"]} & {hindi_capt["CN"]} \\\\ 
            Mandarin  & {mandarin_capt["US"]} & {mandarin_capt["DE"]} & {mandarin_capt["MX"]} & {mandarin_capt["IN"]} & {mandarin_capt["CN"]} \\\\
            """
        elif TABLE_IDX == 2:
            table = f"""
            English   & {english["US"]} & {english["DE"]} & {english["MX"]} & {english["IN"]} & {english["CN"]} \\\\
            German    & {german["US"]} & {german["DE"]} & {german["MX"]} & {german["IN"]} & {german["CN"]} \\\\
            Spanish   & {spanish["US"]} & {spanish["DE"]} & {spanish["MX"]} & {spanish["IN"]} & {spanish["CN"]} \\\\
            Hindi     & {hindi["US"]} & {hindi["DE"]} & {hindi["MX"]} & {hindi["IN"]} & {hindi["CN"]} \\\\
            Mandarin  & {mandarin["US"]} & {mandarin["DE"]} & {mandarin["MX"]} & {mandarin["IN"]} & {mandarin["CN"]} \\\\
            """
        print(table)
    
def extract_llm_answer(response):
    response = response.lower()
    prefixes = [
        "assistant: ", "assistant:", "assistant\n", "assistant",
        "answer choice:", "answer:", "antwort:", "mi respuesta es:", 
        "respuesta:", "respondo:", "选择：", "选项:", "答案:", 
        "答:", "选择 ", "答案：", "选择选项:"
    ]

    for prefix in prefixes:
        if prefix in response:
            response = response.split(prefix)[-1]
    return response

def extract_answer(response):
    response = extract_llm_answer(response)
    response = response.lower()
    response = response.strip()

    b_responses = ["b: hate", "b: non-hate", "answer: b"]
    a_responses = ["a: non-hate", "a: hate", "a) hate", "answer: a"]
    if any(s in response for s in b_responses):
        return "b"
    if any(s in response for s in a_responses):
        return "a"

    response = response.split(" ")[0].split("\n")[0].strip()
    return response 


def check_response(response_llm, response, extracted):
    non_hate_phrases = [
        "this meme is non-hate", "would rate this meme as non-hate", "non-hate (b)", 
        "non-hate.", "no", "option 'a'", "a) non-hate", "not hate.", "not hate speech!", 
        "no, this is not hate speech", "no, the meme is not hate speech", "non-hate", 
        "non-hate:", "गैर नफरत", "sin odio", "नहीं\n\n", "不属于仇恨言论", "不一定是仇恨言论",
        "答案：b：非仇恨", "- क (नॉन-द्वेष)", "नफरत नहीं", "गैर-नफरत", "a：非仇恨", "非仇恨",
        "这个表情包中的内容并未表现出任何攻击或侮辱的情感", "答案：b，非仇恨", "b: 非仇恨", "b. 非仇恨", "a：非仇恨", "b：非仇恨", "b：非仇恨"
    ]
    
    hate_phrases = [
        "this meme contains hate speech", "yes, i would consider this meme as hate speech", 
        "hate", "yes, I would consider this meme as hate speech", "h8", "घृणा", "घृणा",
        "enthält eine hassrede", "enthält eine form von hassre", "hass", "a: odio", "नफरत",
        "इस मीम का उपयोग अपमानजनक", "नफ़रत", "जी हां, यह मीम घृणित भाषण है", "b：仇恨",
    ]
    
    hate_phrases_equal = [
            "yes",
            "hate",
            "h8", "घृणा", "घृणा", "hass",  "odio", "odio (a)", "odio (b)"
        ]
    unable_to_analyze_phrases = [
        "i'm sorry but this task involve", "i cannot analyze", "i'd need to have the meme text translated", 
        "i am unable to interpret the text", "i am unable to provide the analysis", "c", 
        "choose your answer by selecting either 'a' or 'b'", "i am unable to assist", 
        "i'm unable to analyze this meme", "c: unrelated", "i cannot make that determination.", 
        "i can't provide assistance",
        "please provide the meme",
        "das meme enthält eine frage, die", "meme enthält eine aussage",
        "neues jahr!!!", "h.", "मैं नहीं जानता।"
    ]
    unable_to_analyze_phrases_equal = [
        "o", "१६", "घ", "घ्रोणी\n\nअत्याचार"
    ]

    # Check for non-hate phrases
    if any(phrase in response_llm for phrase in non_hate_phrases):
        if "b: Non-hate" in response or "ख: गैर नफरत" in response or "b: Kein Hass" in response or "b: Sin odio" in response:
            return 1
        elif "a: Non-hate" in response or "क: गैर नफरत" in response or "a: Kein Hass" in response or "a: Sin odio" in response:
            return 0
    
    # Check for hate phrases
    if any(phrase in response_llm for phrase in hate_phrases) or any(phrase == response_llm for phrase in hate_phrases_equal):
        if "b: Hate" in response or "ख: घृणा" in response or "b: Hass" in response or "b: Odio" in response:
            return 1
        elif "a: Hate" in response or "क: घृणा" in response or "a: Hass" in response or "a: Odio" in response:
            return 0
    
    # Check for unable to analyze phrases
    if any(phrase in response_llm for phrase in unable_to_analyze_phrases) or any(phrase == response_llm for phrase in unable_to_analyze_phrases_equal):
        return -1

    return -2

def mapping_response(response):
    extracted = extract_answer(response)

    a_response = ["a", "a:", "a: hate", "a.", "a:", "a)", "a (**Non-hate**)", "'a'", "-a-", 
                  "क", "a：仇恨", "अ", "antwort: 'a'", "क:", "क.", "**a**", "答案: a", 
                  "a：非仇恨", "a<|eot_id|>", "का", "答案：a", "答案：a：非仇恨", "回答：a", "答案是：a",
                  "答：a：非仇恨"]
    b_response = ["b", "b:", "b:", "b)", "b:", "b.", "b: hate", "b: non-hate", "(b)", "b：hate", "*b",
                  "ख", "b：非仇恨", "ब", "-b:", "ख:", "答案：b", "b：仇恨", "ख.", "回答：b", "回答：b：非仇恨",
                  "b：", "答：b", "选项b：仇恨", "b:非仇恨", ]

    invalid_response = ["bustin", "busters.kerry", "a.kerry", 
                        "busters.", "bhindiwereview", "bhaving",
                        "bheye.", "africaarabi.", "bilder.", "banner",
                        "banner", "bhaving", "bust", "bubble",
                        "b.19999999", "bietehrt.", "a120.", "ademon", "b20.",
                        "the", "h", "an", "g", "d", "e", "aa", "ग", "n", "keine informationen.",
                        "दोनों में नहीं", "जी-3", "इतिमान से भरा हुआ हो सकता है."]

    # Some more logic:
    
    response_llm = extract_llm_answer(response)
    complex_logic = check_response(response_llm, response, extracted)
    if complex_logic != -2:
        return complex_logic
    """
    # Some more logic:
    response_ = response.split("Assistant: ")[-1]
    if "Not Hate" in response_ or '"non-hate" speech' in response_ or "No, this is not hate speech" in response_ or "This is not hate speech." in response_ or "This meme is non-hate" in response_ or "no." in extracted or "no" == extracted or "non-hate" == extracted or "non-hate." == extracted or "Not hate" == response_ or "Not hate." == response_ or "Not hate speech" == response_ or "Not hate speech." == response_:
        if "b: Non-hate" in response:
            return 1
        elif "a: Non-hate" in response:
            return 0
    if "Hate Speech" in response_ or "hate speech" in response_ or "hate" in extracted or "yes" in extracted or "This meme is hate speech" in response_ or "This meme can be considered Hate (a)" in response_:
        if "b: Hate" in response:
            return 1
        elif "a: Hate" in response:
            return 0
    """
    valid_responses_b = ["choice: b", "meme: b", "选项: b", "答案: b"]

    if extracted in a_response or response_llm == "answer (a)":
        return 0
    elif extracted in b_response or response_llm in valid_responses_b:
        return 1
    elif extracted in invalid_response:
        return -1
    else:
        print(f"Invalid response encountered: {response} \n Extracted: {extracted}")
        return -1


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

def stat_test(df1, df2):
    sample1 = np.array(df1["accuracy"])
    sample2 = np.array(df2["accuracy"])
    p_value = round(ranksums(sample1, sample2).pvalue, 5)
    return p_value


def calc_acc(df, gt_name, predict_name):
    # df = df[df[predict_name] != -1]

    # stat_test(df, gt_name, predict_name)
    # df = df[df["prompt"] == 0]
    accuracy_by_group = calculate_group_accuracy(df, "prompt", gt_name, predict_name)
    # accuracy_by_group = calculate_group_accuracy(df, "ID", gt_name, predict_name)
    # print(accuracy_by_group['accuracy'])
    mean_accuracy = round(accuracy_by_group['accuracy'].mean() * 100, 2)
    std_accuracy = round(np.std(np.array(accuracy_by_group['accuracy']), ddof=1) * 100, 2)
    # print(f"Mean of accuracy & std: {mean_accuracy}, {std_accuracy}")
    # correct_predictions = (df[gt_name] == df[predict_name]).sum()
    # total_predictions = len(df)
    # accuracy = (correct_predictions / total_predictions) * 100
    # print(f"Accuracy for GT Country {gt_name}: {accuracy:.2f}%")

    # Calculate F1 score
    # y_true = list(df[gt_name])
    # y_pred = list(df[predict_name])
    # y_pred = [y_true[i] == 0 if pred == -1 else pred for i, pred in enumerate(y_pred)]
    # f1 = f1_score(y_true, y_pred, average='binary') * 100  # Use 'macro' for multi-class classification

    return mean_accuracy, std_accuracy, accuracy_by_group


if __name__ == '__main__':

    # Open the text file in write mode
    with open('/lustre/project/ki-topml/minbui/repos/HatespeechMemes/output.txt', 'w', encoding='utf-8') as f:
        # Redirect stdout to the file
        sys.stdout = f

        df_gt = process_language_data(ANNOTATION_PATH)
        df_gt = df_gt.reset_index()
        df_gt["ID"] = df_gt["ID"].astype(str)

        latex_preds = {}
        # Loop over all folders inside the parent folder
        for root, dirs, files in os.walk(MODEL_PREDICTIONS):
            for folder in dirs:
                if "archive" in root:
                    continue
                if EVAL_MODELS in folder:
                    latex_preds[folder] = {}
                    print("\n--------------------" + folder + "-------------")
                    for language in LANGUAGES:
                        print("----Language: {}-----".format(language))
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
                        # N Invalid Responses
                        n_invalid = sum(df_inference["hate_prediction"] == -1)
                        latex_preds[folder][language] = {}
                        # Accuracy
                        for language_eval in LANGUAGES:
                            accuracy, std, df_acc = calc_acc(df_inference, MAPPING[language_eval], "hate_prediction")
                            latex_preds[folder][language][MAPPING[language_eval]] = [round(accuracy, 1), round(std, 1), n_invalid, df_acc]


        latex_table(latex_preds)
