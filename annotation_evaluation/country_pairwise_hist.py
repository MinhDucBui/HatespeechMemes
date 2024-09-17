import argparse
import pandas as pd
from itertools import combinations
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import process_language_data


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



def process_language_data(annotation_path, descriptive=False):
    df_languages = {language: [] for language in LANGUAGES}

    for language in LANGUAGES:
        for filter in FILTERS:
            file = os.path.join(
                annotation_path, filter, f"MAIN {language.upper()}_ Cross-Cultural Hate Speech Detection in Memes (Antworten).xlsx"
            )
            df_annotation = pd.read_excel(file)
            ids_all, hate_binary_all, prolific_ids = transform_data_into_pd(
                df_annotation, language)

            zero_counts, one_counts = count_values(hate_binary_all)

            new_dict = {
                'ID': ids_all[0],
                language + "_nonhate_count": zero_counts,
                language + "_hate_count": one_counts
            }
            df = pd.DataFrame(new_dict)
            df["ID"] = df["ID"].astype(str)
            df = df[~df["ID"].isin(SKIP_EXAMPLES)]
            df = df.dropna()
            df_languages[language].append(df)

    for language in LANGUAGES:
        df_languages[language] = pd.concat(df_languages[language])
        df_languages[language] = df_languages[language].groupby(
            'ID').sum().reset_index()
        ties = (df_languages[language][language + '_hate_count']
                == df_languages[language][language + '_nonhate_count'])
        print("Check for ties: ", sum(ties))
        df_languages[language][MAPPING[language]] = (
            df_languages[language][language +
                                   '_hate_count'] > df_languages[language][language + '_nonhate_count']
        ).astype(int)
        if descriptive:
            df_languages[language][MAPPING[language] + "_unanimously"] = \
                (df_languages[language][language + '_hate_count'] == NUM_ANNOTATORS) | \
                (df_languages[language][language + '_nonhate_count'] == NUM_ANNOTATORS)
        df_languages[language].drop(
            columns=[language + '_nonhate_count', language + '_hate_count'], inplace=True)

    # Initialize the merged DataFrame with the first DataFrame in the list
    first_language = LANGUAGES[0]
    merged_df = df_languages[first_language]
    merged_df = merged_df.drop_duplicates()

    # Loop through the remaining DataFrames and merge them
    for language in LANGUAGES[1:]:
        df_languages[language] = df_languages[language].drop_duplicates()
        df_languages[language]["ID"] = df_languages[language]["ID"].astype(str)

        merged_df = pd.merge(
            merged_df, df_languages[language], on='ID', how='left')

    merged_df.set_index('ID', inplace=True)

    merged_df = merged_df.dropna(how="any")

    return merged_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--annotation', '-a', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/prolific_annotations/hatespeech_main/')

    args = parser.parse_args()

    df_languages = {"de": [], "en": [], "zh": [], "es": [], "hi": []}

    merged_df = process_language_data(args.annotation)

    print(merged_df)

    # Check if there are duplicate indices
    duplicate_indices = merged_df.index[merged_df.index.duplicated()]

    # Check if any duplicates exist
    if duplicate_indices.any():
        print("There are duplicate indices in the DataFrame.")
        print("Duplicate index values:")
        print(merged_df.loc[duplicate_indices])
    else:
        print("There are no duplicate indices in the DataFrame.")

    print("Total Number of Data: ", len(merged_df))

    none_count = merged_df.isna().sum().sum()
    print("Number of None values:", none_count)
    df = round(calculate_pairwise_agreement(merged_df), 2)
