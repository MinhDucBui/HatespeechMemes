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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--annotation', '-a', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/prolific_annotations/hatespeech_main/')

    args = parser.parse_args()

    df_languages = {"de": [], "en": [], "zh": [], "es": [], "hi": []}

    merged_df = process_language_data(args.annotation)

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

    print( (df.sum(axis=1) -1)/ 4)

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
    heatmap.set_title('Pairwise Agreement Matrix (%)',
                      fontsize=20)  # Increase title font size
    # heatmap.set_xlabel('Predicted', fontsize=16)  # Increase x-axis label font size
    # heatmap.set_ylabel('Actual', fontsize=16)  # Increase y-axis label font size

    # Adjust the position of the labels to fit nicely
    # Increase x-axis tick label font size
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(rotation=0, fontsize=14)  # Increase y-axis tick label font size

    # Display the plot
    plt.show()
