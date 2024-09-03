import argparse
import pandas as pd
import krippendorff
import re
import os


SKIP_EXAMPLES = ["1134290", "699717", "2061647",
                 "1436", "332838_a", "332838", "6167601"]

DONT_KNOW = None

USER_IDS = "all"

# USER_IDS = ["654a45537577dbb4a1df935a", "65aa84f1aad8b6c3907b39de", "66b65703d01d3e754609455c",
#             "66ccd0c19cae4e65612c78f4", "66859847921f89ca3054d41f", "665f0cf75523dc9152c4af04"]
# USER_IDS = ["65fabcadbea8447f5c937935", "65fb14f7d8a3890c20fac495", "6637725bb8c26344d88019bb"]

import itertools
# combinations = list(itertools.combinations(USER_IDS, 3))
# nested_list = [list(combo) for combo in combinations]
# USER_IDS = nested_list[7]

print(USER_IDS)


LANGUAGES = [
    "hi", "MAIN "
]

FILTERS = ["de", "en", "es", "hi", "zh", "2nd"]
# FILTERS = ["2nd"]

LENGTH = 3


def calculate_0_1s(df):
    # Counting the number of 0s and 1s per row
    count_0s = df.apply(lambda row: (row == 0).sum(), axis=1)
    count_1s = df.apply(lambda row: (row == 1).sum(), axis=1)

    # Adding the counts to the DataFrame
    df['count_0s'] = count_0s
    df['count_1s'] = count_1s
    return df


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
    parser.add_argument('--filter', '-f', type=str,
                        default='')

    args = parser.parse_args()
    filter_file = args.filter
    language = LANGUAGES[0]
    prefix = LANGUAGES[1]

    all_dfs = []
    all_prolifc_ids = []
    for filter in FILTERS:
        file = os.path.join(args.annotation + filter, prefix +
                            f"{language}_ Cross-Cultural Hate Speech Detection in Memes (Antworten).xlsx")

        df_annotation = pd.read_excel(file)

        ids_all = []
        hate_binary_all = []
        prolifc_ids = []
        ids_all, hate_binary_all, prolifc_ids = transform_data_into_pd(
            df_annotation)
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
    print("Total Amount of People: {}".format(len(reliability_data)))
    print("Total Amount of Data: {}".format(len(reliability_data[0])))

    df = df.set_index('ID')

    # Counting the number of 0s and 1s in each column
    count_0s = df.apply(lambda col: (col == 0).sum())
    count_1s = df.apply(lambda col: (col == 1).sum())

    #for column, (zero_count, one_count) in zip(df.columns, zip(count_0s, count_1s)):
    #    print(
    #        f"{column}: 0s = {zero_count}, 1s = {one_count}, -1s = {36-zero_count-one_count}")
    alpha = krippendorff.alpha(
        reliability_data=reliability_data, level_of_measurement="nominal")
    print("\nKrippendorff's alpha: ", alpha)
    print("Num People: ", len(df.keys()))

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
