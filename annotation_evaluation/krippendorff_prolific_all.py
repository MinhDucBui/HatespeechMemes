import argparse
import pandas as pd
import krippendorff
import os
from utils import transform_data_into_pd, process_language_data, SKIP_EXAMPLES, FILTERS
import numpy as np

USER_IDS = "all"

# Spanish: "613ee464768f4be3b5ff3eaa"
#USER_IDS = ["5f3300554188a70deea55136", "662fb34a856a8e0e40696cc9", "5f2ef7740f87702b99055397", 
#            "613f8b830830d428fc93696d", "6577257c0f74c02eed5453c8"
#]


LANGUAGES = [
    "en", "MAIN "
]
# FILTERS = ["zh"]


LENGTH = 5


def calculate_0_1s(df):
    # Counting the number of 0s and 1s per row
    count_0s = df.apply(lambda row: (row == 0).sum(), axis=1)
    count_1s = df.apply(lambda row: (row == 1).sum(), axis=1)

    # Adding the counts to the DataFrame
    df['count_0s'] = count_0s
    df['count_1s'] = count_1s
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--annotation', '-a', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/prolific_annotations/hatespeech_main/')
    parser.add_argument('--filter', '-f', type=str,
                        default='')

    args = parser.parse_args()
    filter_file = args.filter
    language = LANGUAGES[0]
    prefix = LANGUAGES[1]

    all_dfs = []
    all_prolifc_ids = []
    process_language_data(args.annotation)
    for filter in FILTERS:
        file = os.path.join(args.annotation + filter, prefix +
                            f"{language}_ Cross-Cultural Hate Speech Detection in Memes (Antworten).xlsx")

        df_annotation = pd.read_excel(file)

        ids_all = []
        hate_binary_all = []
        prolifc_ids = []
        ids_all, hate_binary_all, prolifc_ids = transform_data_into_pd(
            df_annotation, language)
        # Creating the DataFrame
        new_dict = {
            'ID': ids_all[0],
        }
        for key, hate_binary in enumerate(hate_binary_all):
            new_dict[prolifc_ids[key]] = hate_binary
        df = pd.DataFrame(new_dict)
        df["ID"] = df["ID"].astype(str)

        # df = df[["ID"] + USER_IDS]
        if USER_IDS == "all":
            USER_IDS = prolifc_ids
        all_prolifc_ids = all_prolifc_ids + USER_IDS
        df = df[USER_IDS + ["ID"]]
        all_dfs.append(df)
        USER_IDS = "all"

    USER_IDS = all_prolifc_ids
    df = pd.concat(all_dfs, ignore_index=True)

    df = df.replace({None: np.nan})

    # Identify rows where all specified columns are NaN
    rows_to_drop = df[USER_IDS].isna().all(axis=1)

    # Drop these rows
    df = df[~rows_to_drop]

    df_control = df[df["ID"].isin(SKIP_EXAMPLES)].copy()
    df_control = calculate_0_1s(df_control)
    overlap = sum((df_control['count_0s'] == LENGTH) |
                  (df_control['count_1s'] == LENGTH))

    # Display all columns in a DataFrame
    pd.set_option('display.max_columns', None)
    print(df_control)
    # print("\nControl:\n", overlap)

    # Skip Examples

    df = df[~df["ID"].isin(SKIP_EXAMPLES)]

    df = df.pivot_table(index='ID', aggfunc='first')
    df = df.reset_index()

    reliability_data = []

    # Skip Filter if available
    if os.path.exists(filter_file):
        df_filter = pd.read_csv(filter_file)
        df_filter["Image ID"] = df_filter["Image ID"].astype(int)
        df["ID"] = df["ID"].astype(int)
        df = pd.merge(df, df_filter, left_on='ID',
                      right_on="Image ID", how='left')
        for i in range(2):
            df["hatespeech_" + str(i)] = df["hatespeech"].astype(float)
            USER_IDS.append("hatespeech_" + str(i))
        df = df.drop(columns=['hatespeech', 'Image ID'])

    duplicates = df[df.duplicated(subset='ID', keep=False)]
    print("DUPLICATES: ", len(duplicates["ID"]))
    for index, id in enumerate(USER_IDS):
        reliability_data.append(list(df[id]))
    print("Total Amount of Data: {}".format(len(reliability_data[0])))
    print("Total Amount of Data: {}".format(len(reliability_data[1])))

    df = df.set_index('ID')

    # Counting the number of 0s and 1s in each column
    count_0s = df.apply(lambda col: (col == 0).sum())
    count_1s = df.apply(lambda col: (col == 1).sum())

    # for column, (zero_count, one_count) in zip(df.columns, zip(count_0s, count_1s)):
    #    print(
    #        f"{column}: 0s = {zero_count}, 1s = {one_count}, -1s = {36-zero_count-one_count}")

    alpha = krippendorff.alpha(
        reliability_data=reliability_data, level_of_measurement="nominal")
    print("\nKrippendorff's alpha: ", alpha)
    print("Num People: ", len(set(list(df.keys()))))

    df = calculate_0_1s(df)

    overlap = sum((df['count_0s'] == LENGTH) | (df['count_1s'] == LENGTH))
    total = len(df)
    print("\nPercentage: {}% ({}/{})".format(round(overlap /
                                                   total * 100, 2), overlap, total))
    print("Count of 0s: {}\nCount of 1s: {}".format(
        sum((df['count_0s'] == LENGTH)), sum(df['count_1s'] == LENGTH)))

    hard_2 = sum((df['count_0s'] == 1) & (df['count_1s'] == 2))
    hard_1 = sum((df['count_0s'] == 2) & (df['count_1s'] == 1))
    print("Hard Conflicts: {} & {}".format(hard_1, hard_2))
    soft = sum((df['count_0s'] == 1) & (df['count_1s'] == 1))
    print("Soft Conflicts: {}".format(soft))
    print("All Conflicts: {}".format(hard_2+hard_1+soft))

    USER_IDS = "all"
