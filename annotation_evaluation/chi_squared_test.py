import argparse
import pandas as pd
import re
import os
from collections import Counter
from itertools import combinations
import numpy as np
from scipy.stats import chi2_contingency

LANGUAGES = ["de"]

SKIP_EXAMPLES = {"3208": 0.0,
                 "1436": 1.0,
                 "6453957": 1.0,
                 "332838": 0.0,
                 "2061647": 0.0,
                 "6167601": None
                 }
# Function to calculate pairwise agreement


def perform_chi_squared_test(contingency_table):
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return chi2, p, dof, expected


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--annotation', '-a', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/annotation_evaluation/data')

    args = parser.parse_args()

    dfs = []

    for language in LANGUAGES:
        file = os.path.join(
            args.annotation, f"{language}_ Cross-Cultural Hate Speech Detection in Memes (Antworten).xlsx")
        df_annotation = pd.read_excel(file)
        ids_all = []
        hate_binary_all = []
        prolifc_ids = []
        for index, row in df_annotation.iterrows():
            ids_all.append([])
            hate_binary_all.append([])
            if language == "en":
                prolific_id = row["Please enter your Prolific ID"]
            elif language == "de":
                prolific_id = row["Bitte geben Sie Ihre Prolific-ID ein"]

            #if prolific_id != "65fea00a473f2f7f5070f4d6" and prolific_id != "660e8c48587d881a59230c90":
            prolifc_ids.append(prolific_id)

            for key in df_annotation.keys():
                if language == "en":
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
                if language == "de":
                    if "Bitte geben Sie Ihr Feedback zu Frage" in key:
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

    # Load Characteristics
    dfs = []
    for language in LANGUAGES:
        file = os.path.join(args.annotation, f"{language}_prolific.csv")
        df_annotation = pd.read_csv(file)
        dfs.append(df_annotation)
    df_annotation = pd.concat(dfs)

    # Performing the left join
    result = merged_df.merge(df_annotation, left_on='User ID',
                             right_on='Participant id', how='left')
    result = result[~result["Image ID"].isin(SKIP_EXAMPLES.keys())]

    def create_contingency_table(df, attribute):
        return pd.crosstab(df[attribute], df['hatespeech'])

    #result = result[result["Nationality"] == "Germany"]
    contingency_country = create_contingency_table(result, 'Nationality')
    contingency_user = create_contingency_table(result, 'User ID')
    #contingency_student = create_contingency_table(result, 'Student status')
    print(contingency_country)
    print(contingency_user)

    user_ids = ["660e8c48587d881a59230c90", "65fea00a473f2f7f5070f4d6", "65e0bee6f886849136294ce0"]
    contingency_user = contingency_user[~contingency_user.index.isin(user_ids)]
    print(contingency_user)

    chi2, p_value_country, dof, expected = perform_chi_squared_test(contingency_country)
    chi2, p_value_user, dof, expected = perform_chi_squared_test(contingency_user)
    #p_value_student = perform_chi_squared_test(contingency_student)

    alpha = 0.05

    results = {
        'country': p_value_country,
        'user': p_value_user,
    }

    for attribute, p_value in results.items():
        if p_value < alpha:
            print(
                f'There are significant disparities in annotations across {attribute} (p = {p_value})')
        else:
            print(
                f'There are no significant disparities in annotations across {attribute} (p = {p_value})')
