import argparse
import pandas as pd
import re
import os
from collections import Counter
from itertools import combinations
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


LANGUAGES = ["en", "de", "es", "zh", "hi"]

FILTERS = ["en", "de", "es", "hi", "zh", "2nd"]

DONT_KNOW = None

SKIP_EXAMPLES = ["1134290", "699717", "2061647",
                 "1436", "332838_a", "332838", "6167601"]
# Function to calculate pairwise agreement

MAPPING = {
    "en": "US",
    "de": "DE",
    "es": "MX",
    "zh": "CN",
    "hi": "IN"
}

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
            most_common = count.most_common(2)
            if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
                # There's a tie between the top two elements
                majority_element = None
            elif strict and most_common[0][1] < 3:
                majority_element = None
            else:
                majority_element = most_common[0][0]
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

    df_languages = {"de": [], "en": [], "zh": [], "es": [], "hi": []}

    for language in LANGUAGES:
        for filter in FILTERS:
            file = os.path.join(
                args.annotation, filter, f"MAIN {language}_ Cross-Cultural Hate Speech Detection in Memes (Antworten).xlsx")
            df_annotation = pd.read_excel(file)
            ids_all, hate_binary_all, prolifc_ids = transform_data_into_pd(
                df_annotation)

            result = majority_vote(hate_binary_all, strict=False)
            # for index_user, prolific_id in enumerate(prolifc_ids):
            new_dict = {
                'ID': ids_all[0],
                MAPPING[language]: result
            }
            df = pd.DataFrame(new_dict)
            df = df[~df["ID"].isin(SKIP_EXAMPLES)]
            df = df.dropna()

            df_languages[language].append(df)

    for language in LANGUAGES:
        df_languages[language] = pd.concat(df_languages[language])
    # Initialize the merged DataFrame with the first DataFrame in the list
    first_language = LANGUAGES[0]
    merged_df = df_languages[first_language]
    merged_df = merged_df.drop_duplicates()
    # Loop through the remaining DataFrames and merge them
    for language in LANGUAGES[1:]:
        df_languages[language] = df_languages[language].drop_duplicates()
        merged_df = pd.merge(
            merged_df, df_languages[language], on='ID', how='left')

    merged_df.set_index('ID', inplace=True)

    merged_df = merged_df.dropna(how="any")

    print("Total Number of Data: ", len(merged_df))

    none_count = merged_df.isna().sum().sum()
    print("Number of None values:", none_count)
    df = round(calculate_pairwise_agreement(merged_df), 2)

    # Step 1: Set up the figure size and style
    plt.figure(figsize=(10, 8))
    sns.set(style='whitegrid')

    # Step 2: Create the heatmap
    # Custom colormap
    cmap = sns.color_palette("RdPu", as_cmap=True)

    min_point = df.min().min()
    # Plotting the heatmap
    heatmap = sns.heatmap(df, annot=True, square=True, cmap=cmap, vmin=min_point, vmax=1, 
                        annot_kws={"size": 14})  # Increase annotation font size

    # Customize the plot
    heatmap.set_title('Pairwise Agreement Matrix (%)', fontsize=20)  # Increase title font size
    #heatmap.set_xlabel('Predicted', fontsize=16)  # Increase x-axis label font size
    #heatmap.set_ylabel('Actual', fontsize=16)  # Increase y-axis label font size

    # Adjust the position of the labels to fit nicely
    plt.xticks(rotation=45, ha='right', fontsize=14)  # Increase x-axis tick label font size
    plt.yticks(rotation=0, fontsize=14)  # Increase y-axis tick label font size

    # Display the plot
    # plt.show()
