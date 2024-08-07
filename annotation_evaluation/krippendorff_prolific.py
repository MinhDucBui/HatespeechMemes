import argparse
import pandas as pd
import krippendorff
import re
import os


SKIP_EXAMPLES = ["1134290", "699717", "2061647", "1436", "332838_a", "332838", "6167601"]

# Batch 2
# USER_IDS = ["660e8c48587d881a59230c90", "65fea00a473f2f7f5070f4d6", "65e0bee6f886849136294ce0"]

# Batch 1
USER_IDS = ["65a01f3315177d3861dbdd98",
            "6510335c6c5ba95f95c5af89", "664bc3f078a2df44ca254e10"]

# EN MACE
# USER_IDS = ["5f622ab951e8d7229eced89f", "668f9be093060d7620e0507b", "65f99b502e81655ac7459020", "667aa8193c8a13d292ac9822"]

# DE MACE
# USER_IDS = ["65a6d650253833945cb63d9f", "6429682ff18d08f9eed7d9e7", "6438576b1860de7fc057113d", "627bccc58bb8898efd763beb"]

# NEW DE MACE
USER_IDS = ["580cfe1ceee5dc00017599c3", "62856e7a17891912e6c6fd0a",
            "60f4a86aeb626fcf89cbaf5a", "664512a09c0423e4ddbc89af"]

USER_IDS = "all"

LANGUAGES = [
    "en", "NewMain"
]

LENGTH = 2


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
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/annotation_evaluation/data')
    parser.add_argument('--filter', '-f', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/annotation_evaluation/data/filter/data/NewPRELIMde.csv')

    args = parser.parse_args()
    filter_file = args.filter
    language = LANGUAGES[0]
    prefix = LANGUAGES[1]
    file = os.path.join(args.annotation, prefix +
                        f"{language}_ Cross-Cultural Hate Speech Detection in Memes (Antworten).xlsx")

    df_annotation = pd.read_excel(file)

    ids_all = []
    hate_binary_all = []
    prolifc_ids = []
    for index, row in df_annotation.iterrows():
        ids_all.append([])
        hate_binary_all.append([])
        if language == "en":
            prolifc_ids.append(row["Please enter your Prolific ID"])
        elif language == "de":
            prolifc_ids.append(row["Bitte geben Sie Ihre Prolific-ID ein"])
        for key in df_annotation.keys():
            if language == "en":
                if "Please provide your feedback for Question" in key:
                    id_number = re.search(r'\((\-?\d+)\.jpg\)', key).group(1)
                    if row[key] == "Hate Speech":
                        hate_binary_all[index].append(1)
                    elif row[key] == "Non-Hate Speech":
                        hate_binary_all[index].append(0)
                    else:
                        hate_binary_all[index].append(None)
                    ids_all[index].append(id_number)
            if language == "de":
                if "Bitte geben Sie Ihr Feedback zu Frage" in key:
                    id_number = re.search(r'\((\-?\d+)\.jpg\)', key).group(1)
                    if row[key] == "Hassrede":
                        hate_binary_all[index].append(1)
                    elif row[key] == "Keine Hassrede":
                        hate_binary_all[index].append(0)
                    # elif row[key] == "Ich Weiß Nicht":
                    #    hate_binary_all[index].append(0)
                    else:
                        hate_binary_all[index].append(None)
                    ids_all[index].append(id_number)

    # Creating the DataFrame
    new_dict = {
        'ID': ids_all[0],
    }
    for key, hate_binary in enumerate(hate_binary_all):
        new_dict[prolifc_ids[key]] = hate_binary
    df = pd.DataFrame(new_dict)
    df["ID"] = df["ID"].astype(str)
    # prolifc_ids = prolifc_ids[3:-1]

    # df = df[["ID"] + USER_IDS]
    if USER_IDS == "all":
        USER_IDS = prolifc_ids

    df = df[USER_IDS + ["ID"]]

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
    print("\nControl:\n", overlap)

    # Skip Examples
    df = df[~df["ID"].isin(SKIP_EXAMPLES)]

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

    for index, id in enumerate(USER_IDS):
        reliability_data.append(list(df[id]))

    alpha = krippendorff.alpha(
        reliability_data=reliability_data, level_of_measurement="nominal")
    print("\nKrippendorff's alpha: ", alpha)

    df = df.set_index('ID')

    # Counting the number of 0s and 1s in each column
    count_0s = df.apply(lambda col: (col == 0).sum())
    count_1s = df.apply(lambda col: (col == 1).sum())

    for column, (zero_count, one_count) in zip(df.columns, zip(count_0s, count_1s)):
        print(
            f"{column}: 0s = {zero_count}, 1s = {one_count}, -1s = {36-zero_count-one_count}")

    df = calculate_0_1s(df)

    overlap = sum((df['count_0s'] == LENGTH) | (df['count_1s'] == LENGTH))
    total = len(df)
    print("\nPercentage: {}% ({}/{})".format(round(overlap /
          total * 100, 2), overlap, total))
    print("Count of 0s: {}\nCount of 1s: {}".format(
        sum((df['count_0s'] == LENGTH)), sum(df['count_1s'] == LENGTH)))

    df = df[(df['count_0s'] >= LENGTH) | (df['count_1s'] >= LENGTH)]

    reliability_data = []
    for index, id in enumerate(USER_IDS):
        reliability_data.append(list(df[id]))

    alpha = krippendorff.alpha(
        reliability_data=reliability_data, level_of_measurement="nominal")
    print("\nKrippendorff's alpha: ", alpha)
