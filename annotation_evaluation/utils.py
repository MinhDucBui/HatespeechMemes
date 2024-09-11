import re
import numpy as np
import os
from collections import Counter
import pandas as pd

LANGUAGES = ["en", "de", "es", "zh", "hi"]
FILTERS = ["en", "de", "es", "hi", "zh", "2nd", "3rd_idk"]


DONT_KNOW = None
MAPPING = {
    "en": "US",
    "de": "DE",
    "es": "MX",
    "zh": "CN",
    "hi": "IN"
}
SKIP_EXAMPLES = ["1134290", "699717", "2061647",
                 "1436", "332838_a", "332838", "6167601"]


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


def count_values(hate_binary_all):
    # Count 0s and 1s in each nested list
    hate_binary_all = np.array(hate_binary_all)
    zero_counts = (hate_binary_all == 0).sum(axis=0)
    one_counts = (hate_binary_all == 1).sum(axis=0)
    return zero_counts, one_counts


def transform_data_into_pd(df_annotation, language):
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


def process_language_data(annotation_path):
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
        df_languages[language][MAPPING[language] + "_unanimously"] = \
            (df_languages[language][language + '_hate_count'] == 3) | \
            (df_languages[language][language + '_nonhate_count'] == 3)
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


def transform_data_into_pd_descriptive(df_annotation, language):
    ids_all = []
    hate_binary_all = []
    prolifc_ids = []
    educations = []
    politics = []
    religions = []

    if language == "en" or language == "es" or language == "zh" or language == "hi":
        prolific_key = "Please enter your Prolific ID"
    elif language == "de":
        prolific_key = "Bitte geben Sie Ihre Prolific-ID ein"

    if language == "en" or language == "de":
        education_key = "What is your Level of Education?"
        religion_key = "What is your Religion?"
        political_key = "What is your Political Orientation?"
    elif language == "es":
        education_key = "What is your Level of Education? (¿Cual es tu nivel de educación?)"
        religion_key = "What is your Religion? (¿Cual es tu religión?)"
        political_key = "What is your Political Orientation? (¿Cual es su orientación política?)"
    elif language == "hi":
        education_key = "What is your Level of Education? (आपकी शिक्षा का स्तर क्या है?)"
        religion_key = "What is your Religion? (आपका धर्म क्या है?)"
        political_key = "What is your Political Orientation? (आपका राजनीतिक रुझान क्या है?)"
    elif language == "zh":
        education_key = "What is your Level of Education? (您的教育程度是多少?)"
        religion_key = "What is your Religion? (你的宗教信仰是什么？)"
        political_key = "What is your Political Orientation? (您的政治倾向是什么？)"

    for index, row in df_annotation.iterrows():
        ids_all.append([])
        hate_binary_all.append([])

        prolific_id = row[prolific_key]
        education = row[education_key]
        politic = row[political_key]
        religion = row[religion_key]

        # if prolific_id != "65fea00a473f2f7f5070f4d6" and prolific_id != "660e8c48587d881a59230c90":
        prolifc_ids.append(prolific_id)
        educations.append(education)
        politics.append(politic)
        religions.append(str(religion))

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

    return ids_all, hate_binary_all, prolifc_ids, educations, politics, religions


def process_language_data_descriptive(annotation_path, folder_demo):
    num_annotators = []
    sex = []
    race = []
    education = []
    age = []
    political = []
    religion = []

    # List all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_demo) if f.endswith('.csv')]
    # Load each CSV file and concatenate them into one DataFrame
    df_list = [pd.read_csv(os.path.join(folder_demo, file))
               for file in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset='Participant id')
    for language in LANGUAGES:
        dfs = []
        collect_user_data = []
        for filter in FILTERS:
            file = os.path.join(
                annotation_path, filter, f"MAIN {language}_ Cross-Cultural Hate Speech Detection in Memes (Antworten).xlsx")
            df_annotation = pd.read_excel(file)
            ids_all, hate_binary_all, prolifc_ids, educations, politics, religions = transform_data_into_pd_descriptive(
                df_annotation, language)

            collect_user_data = collect_user_data + \
                [[prolifc_ids[i], educations[i], politics[i], religions[i]]
                    for i in range(len(prolifc_ids))]

            for index_user, prolific_id in enumerate(prolifc_ids):
                new_dict = {
                    'User ID': [prolific_id] * len(hate_binary_all[index_user]),
                    'Image ID': ids_all[0],
                    'Education': educations[0],
                    'Politcal Party': politics[0],
                    "hatespeech": hate_binary_all[index_user]
                }
                df = pd.DataFrame(new_dict)
                df = df[~df["Image ID"].isin(SKIP_EXAMPLES)]
                df["Nationality"] = language
                df = df.dropna()
                dfs.append(df)

        # Concatenate DataFrames
        merged_df = pd.concat(dfs)
        merged_df = merged_df.dropna()
        # result = merged_df.groupby(['User ID']).value_counts()
        # user_id = list(set(merged_df["User ID"]))
        df_user = pd.DataFrame(collect_user_data, columns=[
                               'User ID', 'Education', 'Political', 'Religion'])

        df_demo = pd.merge(df_user, combined_df, left_on='User ID',
                           right_on="Participant id", how='left')
        df_demo = df_demo.drop_duplicates(subset=['User ID'])

        # Collect Attributes
        user_ids = df_demo["User ID"]
        num_annotators.append(len(set(user_ids)))
        sex.append(df_demo['Sex'].value_counts())
        race.append(df_demo['Ethnicity simplified'].value_counts())
        education.append(df_demo['Education'].value_counts())

        bins = [18, 19, 29, 39, 49, 59, 69, 79, 89]
        labels = ['18-19', '20-29', '30-39', '40-49',
                  '50-59', '60-69', '70-79', '80-89']
        pd.set_option('display.max_columns', None)

        df_demo['Age'] = df_demo['Age'].astype(int)
        df_demo['Age Group'] = pd.cut(
            df_demo['Age'], bins=bins, labels=labels, right=True, include_lowest=True)
        age.append(df_demo['Age Group'].value_counts())
        political.append(df_demo['Political'].value_counts(dropna=False))
        religion.append(df_demo['Religion'].value_counts())

    return num_annotators, sex, race, education, age, political, religion
