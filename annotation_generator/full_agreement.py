import argparse
import pandas as pd
import re
import os
import numpy as np
import krippendorff

DONT_KNOW = -1

SKIP_EXAMPLES = {"1134290": 0.0,
                 "699717": 1.0,
                 "2061647": 1.0,
                 "1436": 0.0,
                 "332838_a": 0.0,
                 "332838": 1.0,
                 "6167601": 0.0
                 }
#SKIP_EXAMPLES = ["1134290", "699717", "2061647", "1436", "332838_a", "332838", "6167601"]

LANGUAGES = {
    "es": "PRELIMINARY "
}

# USER_IDS = ["663bc6326df9a16437a08660", "6631e783e321299d4d737ec2", "655f017761462616d393ed61", "660afb99816308580be88c56"]
USER_IDS = "all"


def find_threshold(data, percentage):
    # Sort the list
    sorted_data = sorted(data)

    # Calculate the index for the threshold
    index = int(len(sorted_data) * percentage / 100)

    # Ensure index is within bounds
    index = min(index, len(sorted_data) - 1)

    # Get the threshold value
    threshold = sorted_data[index]

    return threshold


def transform_data_into_pd(df_annotation):
    ids_all = []
    hate_binary_all = []
    prolifc_ids = []
    for index, row in df_annotation.iterrows():
        ids_all.append([])
        hate_binary_all.append([])
        if "Please enter your Prolific ID" in row.keys():
            prolific_id = row["Please enter your Prolific ID"]
        elif "Bitte geben Sie Ihre Prolific-ID ein" in row.keys():
            prolific_id = row["Bitte geben Sie Ihre Prolific-ID ein"]

        # if prolific_id != "65fea00a473f2f7f5070f4d6" and prolific_id != "660e8c48587d881a59230c90":
        prolifc_ids.append(prolific_id)

        for key in df_annotation.keys():
            if "Please provide your feedback for Question" in key:
                id_number = re.search(
                    r'\((\-?\d+)\.jpg\)', key).group(1)
                if row[key] == "Hate Speech":
                    hate_binary_all[index].append(1)
                elif row[key] == "Non-Hate Speech":
                    hate_binary_all[index].append(0)
                elif row[key] == "I Don't Know":
                    hate_binary_all[index].append(DONT_KNOW)
                else:
                    hate_binary_all[index].append(None)
                ids_all[index].append(id_number)
            elif "Bitte geben Sie Ihr Feedback zu Frage" in key:
                id_number = re.search(
                    r'\((\-?\d+)\.jpg\)', key).group(1)
                if row[key] == "Hassrede":
                    hate_binary_all[index].append(1)
                elif row[key] == "Keine Hassrede":
                    hate_binary_all[index].append(0)
                elif row[key] == "Ich Weiß Nicht":
                    hate_binary_all[index].append(DONT_KNOW)
                else:
                    hate_binary_all[index].append(None)
                ids_all[index].append(id_number)
            if language == "zh":
                if "请提供您对问题的反馈" in key:
                    id_number = re.search(r'\((\-?\d+)\.jpg\)', key).group(1)
                    if row[key] == "仇恨言论":
                        hate_binary_all[index].append(1)
                    elif row[key] == "非仇恨言论":
                        hate_binary_all[index].append(0)
                    # elif row[key] == "Ich Weiß Nicht":
                    #    hate_binary_all[index].append(0)
                    else:
                        hate_binary_all[index].append(None)
                    ids_all[index].append(id_number)

            if language == "hi":
                if "कृपया प्रश्न के लिए अपनी प्रतिक्रिया" in key:
                    id_number = re.search(r'\((\-?\d+)\.jpg\)', key).group(1)
                    if row[key] == "घृणास्पद भाषण है":
                        hate_binary_all[index].append(1)
                    elif row[key] == "गैर-घृणास्पद भाषण":
                        hate_binary_all[index].append(0)
                    # elif row[key] == "Ich Weiß Nicht":
                    #    hate_binary_all[index].append(0)
                    else:
                        hate_binary_all[index].append(None)
                    ids_all[index].append(id_number)

            if language == "es":
                if "Por favor, proporcione sus comentarios sobre la pregunta" in key:
                    id_number = re.search(r'\((\-?\d+)\.jpg\)', key).group(1)
                    if row[key] == "Discurso de odio":
                        hate_binary_all[index].append(1)
                    elif row[key] == "Discurso sin odio":
                        hate_binary_all[index].append(0)
                    # elif row[key] == "Ich Weiß Nicht":
                    #    hate_binary_all[index].append(0)
                    else:
                        hate_binary_all[index].append(None)
                    ids_all[index].append(id_number)

    return ids_all, hate_binary_all, prolifc_ids


def check_all_equal(lst):
    # Convert each inner list to a tuple and check if all tuples are equal
    first = tuple(lst[0])
    return all(tuple(inner_list) == first for inner_list in lst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--output_folder', '-o', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/prolific_annotations/filter_full_aggreement')
    parser.add_argument('--annotation', '-a', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/prolific_annotations/hatespeech_prelim')

    args = parser.parse_args()

    diff_df = {}

    for language, file_prefix in LANGUAGES.items():
        dfs = []
        file = os.path.join(
            args.annotation, file_prefix + f"{language}_ Cross-Cultural Hate Speech Detection in Memes (Antworten).xlsx")
        if os.path.isfile(file):
            df_annotation = pd.read_excel(file)
        else:
            continue

        ids_all, hate_binary_all, prolifc_ids = transform_data_into_pd(
            df_annotation)

        # Given data
        hate_binary_all = list(zip(*hate_binary_all))

        # Create the DataFrame from the list of dictionaries
        df = pd.DataFrame(data=hate_binary_all, columns=prolifc_ids)
        if check_all_equal(ids_all):
            df["Image ID"] = ids_all[0]
        else:
            raise "Error"

        if USER_IDS != "all":
            df = df[USER_IDS + ["Image ID"]]
        else:
            USER_IDS = prolifc_ids

        rows_to_keep = df.drop(columns=['Image ID']).isna().all(axis=1)
        df = df[~rows_to_keep]
        #df = df.drop(columns=REMOVE_IDS)
        # Remove Skip Examples
        df = df[~df["Image ID"].isin(SKIP_EXAMPLES.keys())]
        non_hate_count = df.apply(lambda row: (row == 0).sum(), axis=1)
        hate_count = df.apply(lambda row: (row == 1).sum(), axis=1)
        idk_count = df.apply(lambda row: (row == -1).sum(), axis=1)
        df["non_hate_count"] = non_hate_count
        df["hate_count"] = hate_count
        df["idk_count"] = idk_count
        df["total_valid_count"] = non_hate_count + hate_count + idk_count

        df_subset = df[(df["non_hate_count"] == df["total_valid_count"]) | (
            df["hate_count"] == df["total_valid_count"])].copy()
        df_subset['hatespeech'] = df_subset['hate_count'] == df_subset["total_valid_count"]
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.expand_frame_repr', False)  # Do not wrap the DataFrame display

        reliability_data = []
        for index, id in enumerate(USER_IDS):
            #if id in REMOVE_IDS:
            #    continue
            reliability_data.append(list(df_subset[id]))

        alpha = krippendorff.alpha(
            reliability_data=reliability_data, level_of_measurement="nominal")
        print("Krippendorff's alpha: ", alpha)

        print(len(df_subset))
        df_subset = df_subset[["Image ID", "hatespeech"]]

        # Saving the DataFrame to a CSV file
        output_file = os.path.join(
            args.output_folder, file_prefix + f"{language}.csv")
        df_subset.to_csv(output_file, index=False)
