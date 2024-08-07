import argparse
import pandas as pd
import re
import os
from collections import Counter
from itertools import combinations
import numpy as np


LANGUAGES = {"de": "NewMain",
             "en": "NewMain"}
LANGUAGES_FILTER = {"de": "NewPrelim"}
DONT_KNOW = None
SKIP_EXAMPLES = ["1134290", "699717", "2061647",
                 "1436", "332838_a", "332838", "6167601"]
# Function to calculate pairwise agreement


def calculate_pairwise_agreement(df):
    annotators = df.columns
    n_annotators = len(annotators)
    agreement_matrix = np.ones((n_annotators, n_annotators))

    for (i, annotator_1), (j, annotator_2) in combinations(enumerate(annotators), 2):
        matches = (df[annotator_1] == df[annotator_2]).sum()
        total = len(df)
        agreement = matches / total
        agreement_matrix[i, j] = agreement
        agreement_matrix[j, i] = agreement  # Symmetric matrix

    # Convert matrix to DataFrame for better readability
    agreement_df = pd.DataFrame(
        agreement_matrix, index=annotators, columns=annotators)
    return agreement_df


def majority_vote(lists, strict=True):
    # Transpose the list of lists
    transposed_lists = list(zip(*lists))

    # Determine the majority element at each position
    majority_list = []
    for position in transposed_lists:
        # Filter out None values
        filtered_position = [elem for elem in position if elem is not None]

        if filtered_position:
            count = Counter(filtered_position)
            if strict and count.most_common(1)[0][1] < 3:
                majority_element = None
            else:
                majority_element = count.most_common(1)[0][0]
        else:
            majority_element = None

        majority_list.append(majority_element)

    return majority_list


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

    return ids_all, hate_binary_all, prolifc_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--annotation', '-a', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/annotation_evaluation/data')

    args = parser.parse_args()

    df_languages = {}

    for language, prefix in LANGUAGES.items():
        file = os.path.join(
            args.annotation, prefix + f"{language}_ Cross-Cultural Hate Speech Detection in Memes (Antworten).xlsx")
        df_annotation = pd.read_excel(file)
        ids_all, hate_binary_all, prolifc_ids = transform_data_into_pd(
            df_annotation)

        result = majority_vote(hate_binary_all, strict=True)
        new_dict = {
            'ID': ids_all[0],
            "hatespeech_" + str(language): result
        }
        df = pd.DataFrame(new_dict)
        df = df[~df["ID"].isin(SKIP_EXAMPLES)]
        df = df.dropna()
        image_ids = list(pd.concat(df)["Image ID"].unique())
        # Filter
        for language, prefix in LANGUAGES_FILTER.items():
            file = os.path.join(
                args.annotation, prefix + f"{language}_ Cross-Cultural Hate Speech Detection in Memes (Antworten).xlsx")
            df_annotation = pd.read_excel(file)
            ids_all, hate_binary_all, prolifc_ids = transform_data_into_pd(
                df_annotation)
            result = majority_vote(hate_binary_all, strict=True)
            new_dict = {
                'ID': ids_all[0],
                "hatespeech_" + str(language): result
            }
            df_filter = pd.DataFrame(new_dict)
            df_filter = df_filter[~df_filter["ID"].isin(SKIP_EXAMPLES)]
            df_filter = df_filter.dropna()

        df_languages[language] = df

    # Initialize the merged DataFrame with the first DataFrame in the list
    first_language = list(LANGUAGES.keys())[0]
    merged_df = df_languages[first_language]

    # Loop through the remaining DataFrames and merge them
    for language in list(LANGUAGES.keys())[1:]:
        merged_df = pd.merge(
            merged_df, df_languages[language], on='ID', how='outer')

    merged_df.set_index('ID', inplace=True)
    merged_df = merged_df.dropna(how='any')
    print(merged_df)
    df = calculate_pairwise_agreement(merged_df)
    print(df)
