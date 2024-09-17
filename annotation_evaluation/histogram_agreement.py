import argparse
from utils import process_language_data
import matplotlib.pyplot as plt
import pandas as pd
import random

NUM_EXAMPLES = 15
random.seed(42)  # You can use any integer as the seed


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--annotation', '-a', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/prolific_annotations/hatespeech_main/')

    args = parser.parse_args()
    df = process_language_data(args.annotation)

    # Load Category Mapping
    df_category_mapping = pd.read_csv("/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/annotation_evaluation/category_mapping.csv")

    # Count how many instances we have per category:
    df_count = df.reset_index()
    df_count["ID"] = df_count["ID"].astype(str)
    df_category_mapping["Instance"] = df_category_mapping["Instance"].astype(str)

    df_count = pd.merge(df_category_mapping, df_count, left_on="Instance", right_on="ID", how="inner")

    subcategory_counts = df_count["Subcategory"].value_counts()
    print(df_count["Category"].value_counts())
    # dfasd
    # Map the subcategory counts back to the DataFrame
    df_count['Subcategory_Count'] = df_count['Subcategory'].map(subcategory_counts)

    # Sort the DataFrame based on the subcategory count
    df_sorted = df_count.sort_values(by='Category', ascending=False)
    df_sorted = df_sorted[["Category", "Subcategory", "Subcategory_Count"]].drop_duplicates()
    print(df_sorted)

    # Initialize counters
    full_agreements = 0
    full_minus_1_agreements = 0
    full_minus_2_agreements = 0

    ids = [[], [], []]

    # Iterate over each row, access the index separately
    for row in df.itertuples(index=True, name=None):
        index = row[0]  # Collect the index
        values = row[1:]  # Collect the values, excluding the index
        unique_values = pd.Series(values).value_counts()
        if len(unique_values) == 1:
            full_agreements += 1
            ids[0].append([index, values])
        elif len(unique_values) == 2 and 1 in unique_values.values:
            full_minus_1_agreements += 1
            ids[1].append([index, values])
        elif len(unique_values) == 2 and 2 in unique_values.values:
            full_minus_2_agreements += 1
            ids[2].append([index, values])

    # Print out Examples:
    random.shuffle(ids[0])
    random.shuffle(ids[1])
    random.shuffle(ids[2])
    print("\n------Full Agreement IDs:------")
    print(df.keys())
    for id, hatespeech_binary in ids[0][:NUM_EXAMPLES]:
        print("Image ID: {} with Hatespeech: {}".format(id, hatespeech_binary))
    print("\n--------4 Countries Agree IDs:--------")
    for id, hatespeech_binary in ids[1][:NUM_EXAMPLES]:
        print("Image ID: {} with Hatespeech: {}".format(id, hatespeech_binary))
    print("\n--------3 Countries Agree IDs:--------")
    for id, hatespeech_binary in ids[2][:NUM_EXAMPLES]:
        print("Image ID: {} with Hatespeech: {}".format(id, hatespeech_binary))

    # Create a prettier and larger histogram
    agreement_counts = {
        'Full Agreement': full_agreements,
        '4 Countries Agree': full_minus_1_agreements,
        '3 Countries Agree': full_minus_2_agreements
    }

    # Set the figure size to make the plot bigger
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the width and height

    # Customize colors and bars
    bars = ax.barh(list(agreement_counts.keys()), list(
        agreement_counts.values()), color=['#66c2a5', '#fc8d62', '#8da0cb'])

    # Add gridlines
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Add bar values (counts) on the bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height() /
                2, f'{int(width)}', va='center', fontsize=12)

    # Customize titles and labels
    ax.set_title('Agreement Types Histogram', fontsize=16, weight='bold')
    ax.set_xlabel('Count', fontsize=12)
    ax.set_ylabel('Agreement Type', fontsize=12)

    # Show the plot
    plt.tight_layout()
    plt.show()
