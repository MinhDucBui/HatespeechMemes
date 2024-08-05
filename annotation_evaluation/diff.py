import argparse
import pandas as pd
import re
import os
from collections import Counter
from itertools import combinations
import numpy as np
from scipy.stats import chi2_contingency


LANGUAGES = {
    "en": "OLD_",
    "de": "OLD_"
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--annotation', '-a', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/annotation_evaluation/data')

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
                        hate_binary_all[index].append(-1)
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
                        hate_binary_all[index].append(-1)
                    else:
                        hate_binary_all[index].append(None)
                    ids_all[index].append(id_number)
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
        grouped = merged_df.groupby("Image ID")["hatespeech"].value_counts()

        enough = []
        not_enough = []
        diff_df[language] = []
        # Iterate over the grouped object and print the results in pairs
        for image_id, counts in grouped.groupby(level=0):
            total_count = 0
            hatespeech_dict = {}
            for hatespeech_value, count in counts.items():
                total_count += count
                hatespeech_dict[int(hatespeech_value[1])] = count
            max_key = max(hatespeech_dict, key=hatespeech_dict.get)
            diff_df[language].append(
                [hatespeech_value[0], max_key, hatespeech_dict[max_key], total_count])

    columns = ["ID", "Hatespeech", "count", "total_count"]
    print(diff_df)
    en_df = pd.DataFrame(diff_df["en"], columns=[
                         "en_" + column for column in columns])
    de_df = pd.DataFrame(diff_df["de"], columns=[
                         "de_" + column for column in columns])

    inner_join_df = pd.merge(en_df, de_df, left_on='en_ID', right_on='de_ID')
    inner_join_df = inner_join_df[inner_join_df["de_Hatespeech"] != -1]
    inner_join_df["diff"] = inner_join_df["en_Hatespeech"] == inner_join_df["de_Hatespeech"]
    print("\nOverlap in Percentage: ", sum(
        inner_join_df["diff"]) / len(inner_join_df) * 100)
    diff_df = inner_join_df[(inner_join_df["en_count"] == inner_join_df["en_total_count"]) & (
        inner_join_df["de_count"] == (inner_join_df["de_total_count"]-1))]
    print(diff_df)
    diff_df = inner_join_df[(inner_join_df["en_count"] == inner_join_df["en_total_count"]) & (
        inner_join_df["de_count"] == (inner_join_df["de_total_count"]))]
    print(diff_df)
