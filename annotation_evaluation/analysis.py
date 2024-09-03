import argparse
import pandas as pd
import re
import os
from collections import Counter
from itertools import combinations
import numpy as np
from scipy.stats import chi2_contingency


SKIP_EXAMPLES = ["1134290", "699717", "2061647",
                 "1436", "332838_a", "332838", "6167601"]

DONT_KNOW = None

LANGUAGE = "hi"
FILTERS = ["en", "de", "es"]


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

            elif language == "zh":
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

            elif language == "hi":
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

            elif language == "es":
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--annotation', '-a', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/prolific_annotations/hatespeech_main/')

    args = parser.parse_args()
    language = LANGUAGE

    dfs = []

    for filter in FILTERS:
        file = os.path.join(
            args.annotation + filter, "MAIN " + f"{language}_ Cross-Cultural Hate Speech Detection in Memes (Antworten).xlsx")
        df_annotation = pd.read_excel(file)
        ids_all = []
        hate_binary_all = []
        prolifc_ids = []
        ids_all, hate_binary_all, prolifc_ids = transform_data_into_pd(
            df_annotation)

        for index_user, prolific_id in enumerate(prolifc_ids):
            new_dict = {
                'User ID': [prolific_id] * len(hate_binary_all[index_user]),
                'Image ID': ids_all[0],
                "hatespeech": hate_binary_all[index_user]
            }
            df = pd.DataFrame(new_dict)
            dfs.append(df)

    # Concatenate DataFrames
    merged_df = pd.concat(dfs)
    merged_df = merged_df.dropna()

    # Remove Skip Examples
    merged_df = merged_df[~merged_df["Image ID"].isin(SKIP_EXAMPLES)]
    grouped = merged_df.groupby("Image ID")["hatespeech"].value_counts()
    result = merged_df.groupby('Image ID')['hatespeech'].agg(lambda x: x.mode()[0])

    print(result.value_counts() / 145)

    """
    print(most_common_value)
    print(merged_df["hatespeech"].value_counts())

    hatespeech_aggregated = {}
    # Iterate over the grouped object and print the results in pairs
    for image_id, counts in grouped.groupby(level=0):
        total_count = 0
        hatespeech_dict = {}
        for hatespeech_value, count in counts.items():
            total_count += count
            hatespeech_dict[int(hatespeech_value[1])] = count

        max_key = max(hatespeech_dict, key=hatespeech_dict.get)
        hatespeech_aggregated[image_id] = max_key
        if hatespeech_dict[max_key] < total_count:
            not_enough.append(
                f'    Hatespeech: {hatespeech_value}, Count: {hatespeech_dict[max_key]}, Total: {total_count}')
        else:
            enough.append(
                f'    Hatespeech: {hatespeech_value}, Count: {hatespeech_dict[max_key]}, Total: {total_count}')

    pd.DataFrame(hatespeech_aggregated)
    print(hatespeech_aggregated)
    """