import argparse
import pandas as pd
import re
import os
from collections import Counter
from itertools import combinations
import numpy as np
from scipy.stats import chi2_contingency

LANGUAGES = ["en", "de"]

FILTERS = ["en", "de", "es"]


SKIP_EXAMPLES = ["1134290", "699717", "2061647",
                 "1436", "332838_a", "332838", "6167601"]

DONT_KNOW = None

def fill_latex_table(num_annotators, sex, race, education, age, political, religion):
    countries = ['USA', 'Germany', 'Mexico', 'India', 'China']

    template = r"""
    \begin{{table*}}
    \centering
    \small
    \setlength{{\tabcolsep}}{{5pt}}
    \begin{{tabular}}{{l|ccccc}}
    \toprule
    & USA & Germany & Mexico & India & China  \\
    \midrule
    \textbf{{No. of Annotators}} & {annotators[0]} & {annotators[1]} & {annotators[2]} & {annotators[3]} & {annotators[4]} \\ \midrule
    \textbf{{Gender}} (\%) &  &  &  &  &  \\
    \quad male & {male[0]} & {male[1]} & {male[2]} & {male[3]} & {male[4]} \\ 
    \quad female & {female[0]} & {female[1]} & {female[2]} & {female[3]} & {female[4]} \\ \midrule
    \textbf{{Race}} (\%) &  &  &  &  &  \\
    \quad Asian & {asian[0]} & {asian[1]} & {asian[2]} & {asian[3]} & {asian[4]} \\ 
    \quad Black & {black[0]} & {black[1]} & {black[2]} & {black[3]} & {black[4]} \\ 
    \quad Hispanic & {hispanic[0]} & {hispanic[1]} & {hispanic[2]} & {hispanic[3]} & {hispanic[4]} \\ 
    \quad Middle Eastern & {middle_eastern[0]} & {middle_eastern[1]} & {middle_eastern[2]} & {middle_eastern[3]} & {middle_eastern[4]} \\ 
    \quad White & {white[0]} & {white[1]} & {white[2]} & {white[3]} & {white[4]} \\ 
    \quad Other & {other_race[0]} & {other_race[1]} & {other_race[2]} & {other_race[3]} & {other_race[4]} \\ \midrule
    \textbf{{Level of Education}} (\%) &  &  &  &  &  \\ 
    \quad Below High School & {below_high_school[0]} & {below_high_school[1]} & {below_high_school[2]} & {below_high_school[3]} & {below_high_school[4]} \\ 
    \quad High School & {high_school[0]} & {high_school[1]} & {high_school[2]} & {high_school[3]} & {high_school[4]} \\ 
    \quad College & {college[0]} & {college[1]} & {college[2]} & {college[3]} & {college[4]} \\ 
    \quad Bachelor & {bachelor[0]} & {bachelor[1]} & {bachelor[2]} & {bachelor[3]} & {bachelor[4]} \\ 
    \quad Master’s Degree & {masters[0]} & {masters[1]} & {masters[2]} & {masters[3]} & {masters[4]} \\ 
    \quad Doctorate & {doctorate[0]} & {doctorate[1]} & {doctorate[2]} & {doctorate[3]} & {doctorate[4]} \\ \midrule
    \textbf{{Age}} (\%) &  &  &  &  &  \\ 
    \quad 18-19 & {age_18_19[0]} & {age_18_19[1]} & {age_18_19[2]} & {age_18_19[3]} & {age_18_19[4]} \\ 
    \quad 20-29 & {age_20_29[0]} & {age_20_29[1]} & {age_20_29[2]} & {age_20_29[3]} & {age_20_29[4]} \\ 
    \quad 30-39 & {age_30_39[0]} & {age_30_39[1]} & {age_30_39[2]} & {age_30_39[3]} & {age_30_39[4]} \\ 
    \quad 40-49 & {age_40_49[0]} & {age_40_49[1]} & {age_40_49[2]} & {age_40_49[3]} & {age_40_49[4]} \\ 
    \quad 50-59 & {age_50_59[0]} & {age_50_59[1]} & {age_50_59[2]} & {age_50_59[3]} & {age_50_59[4]} \\ 
    \quad 60-69 & {age_60_69[0]} & {age_60_69[1]} & {age_60_69[2]} & {age_60_69[3]} & {age_60_69[4]} \\ 
    \quad 70-79 & {age_70_79[0]} & {age_70_79[1]} & {age_70_79[2]} & {age_70_79[3]} & {age_70_79[4]} \\ 
    \quad 80-89 & {age_80_89[0]} & {age_80_89[1]} & {age_80_89[2]} & {age_80_89[3]} & {age_80_89[4]} \\ \midrule
    \textbf{{Political Orientation}} (\%) &  &  &  &  &  \\ 
    \quad Liberal/Progressive & {liberal_progressive[0]} & {liberal_progressive[1]} & {liberal_progressive[2]} & {liberal_progressive[3]} & {liberal_progressive[4]} \\ 
    \quad Moderate Liberal & {moderate_liberal[0]} & {moderate_liberal[1]} & {moderate_liberal[2]} & {moderate_liberal[3]} & {moderate_liberal[4]} \\ 
    \quad Independent & {independent[0]} & {independent[1]} & {independent[2]} & {independent[3]} & {independent[4]} \\ 
    \quad Moderate Conservative & {moderate_conservative[0]} & {moderate_conservative[1]} & {moderate_conservative[2]} & {moderate_conservative[3]} & {moderate_conservative[4]} \\ 
    \quad Conservative & {conservative[0]} & {conservative[1]} & {conservative[2]} & {conservative[3]} & {conservative[4]} \\ \midrule
    \textbf{{Religion}} (\%) &  &  &  &  &  \\ 
    \quad None & {none_religion[0]} & {none_religion[1]} & {none_religion[2]} & {none_religion[3]} & {none_religion[4]} \\ 
    \quad Christian & {christian[0]} & {christian[1]} & {christian[2]} & {christian[3]} & {christian[4]} \\ 
    \quad Buddhism & {buddhism[0]} & {buddhism[1]} & {buddhism[2]} & {buddhism[3]} & {buddhism[4]} \\ 
    \quad Islam & {islam[0]} & {islam[1]} & {islam[2]} & {islam[3]} & {islam[4]} \\ 
    \quad Judaism & {judaism[0]} & {judaism[1]} & {judaism[2]} & {judaism[3]} & {judaism[4]} \\ 
    \quad Hinduism & {hinduism[0]} & {hinduism[1]} & {hinduism[2]} & {hinduism[3]} & {hinduism[4]} \\ 
    \quad Irreligion & {irreligion[0]} & {irreligion[1]} & {irreligion[2]} & {irreligion[3]} & {irreligion[4]} \\ 
    \quad Other & {other_religion[0]} & {other_religion[1]} & {other_religion[2]} & {other_religion[3]} & {other_religion[4]} \\ \midrule
    \bottomrule
    \end{{tabular}}
    \caption{{Ablation study on the development set for \texttt{{(4) Mixtrals}}, our best system.}}
    \label{{table:ablation}}
    \end{{table*}}
    """

    # Sex
    male = [df["Male"] for df in sex]
    female = [df["Female"] for df in sex]
    print(male)

    # Race
    print(race)
    asian = [df["Asian"] if "Asian" in df else "--" for df in race]
    black = [df["Black"] if "Black" in df else "--" for df in race]
    hispanic = [df["Hispanic"] if "Hispanic" in df else "--" for df in race]
    middle = [df["Middle Eastern"] if "Middle Eastern" in df else "--" for df in race]
    white = [df["White"] if "White" in df else "--" for df in race]
    other = [df["Other"] if "Other" in df else "--" for df in race]
    das

 
    filled_template = template.format(
        annotators=num_annotators,
        male=male,
        female=female,
        asian=asian,
        black=black,
        hispanic=hispanic,
        middle_eastern=middle,
        white=white,
        other_race=other,
        below_high_school=table_values[country],
        high_school=table_values[country],
        college=table_values[country],
        bachelor=table_values[country],
        masters=table_values[country],
        doctorate=table_values[country],
        age_18_19=table_values[country],
        age_20_29=table_values[country],
        age_30_39=table_values[country],
        age_40_49=table_values[country],
        age_50_59=table_values[country],
        age_60_69=table_values[country],
        age_70_79=table_values[country],
        age_80_89=table_values[country],
        liberal_progressive=table_values[country],
        moderate_liberal=table_values[country],
        independent=table_values[country],
        moderate_conservative=table_values[country],
        conservative=table_values[country],
        none_religion=table_values[country],
        christian=table_values[country],
        buddhism=table_values[country],
        islam=table_values[country],
        judaism=table_values[country],
        hinduism=table_values[country],
        irreligion=table_values[country],
        other_religion=table_values[country],
    )
    print(filled_template)
    return filled_template



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


def transform_data_into_pd(df_annotation):
    ids_all = []
    hate_binary_all = []
    prolifc_ids = []
    educations = []
    politics = []
    religions = []
    for index, row in df_annotation.iterrows():
        ids_all.append([])
        hate_binary_all.append([])
        if "Please enter your Prolific ID" in row.keys():
            prolific_id = row["Please enter your Prolific ID"]
        elif "Bitte geben Sie Ihre Prolific-ID ein" in row.keys():
            prolific_id = row["Bitte geben Sie Ihre Prolific-ID ein"]
        print(row)
        education = row["What is your Level of Education?"]

        politic = row["What is your Political Orientation?"]

        religion = row["What is your Religion?"]

        # if prolific_id != "65fea00a473f2f7f5070f4d6" and prolific_id != "660e8c48587d881a59230c90":
        prolifc_ids.append(prolific_id)
        educations.append(education)
        politics.append(politic)
        religions.append(religion)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--annotation', '-a', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/prolific_annotations/hatespeech_main/')
    parser.add_argument('--demo', '-d', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/prolific_annotations/annotator_demo')

    args = parser.parse_args()

    num_annotators = []
    sex = []
    race = []
    education = []
    age = []
    political = []
    religion = []

    for language in LANGUAGES:
        dfs = []
        collect_user_data = []
        for filter in FILTERS:
            file = os.path.join(
                args.annotation, filter, f"MAIN {language}_ Cross-Cultural Hate Speech Detection in Memes (Antworten).xlsx")
            df_annotation = pd.read_excel(file)
            ids_all, hate_binary_all, prolifc_ids, educations, politics, religions = transform_data_into_pd(
                df_annotation)

            collect_user_data = collect_user_data + [[prolifc_ids[i], educations[i], politics[i], religions[i]] for i in range(len(prolifc_ids))]

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
        # print(result)
        # user_id = list(set(merged_df["User ID"]))
        df_user = pd.DataFrame(collect_user_data, columns=['User ID', 'Education', 'Political', 'Religion'])

        # Load demographics
        folder_demo = args.demo
        # List all CSV files in the folder
        csv_files = [f for f in os.listdir(folder_demo) if f.endswith('.csv')]

        # Load each CSV file and concatenate them into one DataFrame
        df_list = [pd.read_csv(os.path.join(folder_demo, file))
                    for file in csv_files]
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset='Participant id')

        df_demo = pd.merge(df_user, combined_df, left_on='User ID',
                            right_on="Participant id", how='left')

        # Collect Attributes
        num_annotators.append(len(df_demo))
        sex.append(df_demo['Sex'].value_counts())
        race.append(df_demo['Ethnicity simplified'].value_counts())
        education.append(df_demo['Education'].value_counts())

        bins = [18, 19, 29, 39, 49, 59, 69, 79, 89]
        labels = ['18-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89']
        df_demo['Age'] = df_demo['Age'].astype(int)
        df_demo['Age Group'] = pd.cut(df_demo['Age'], bins=bins, labels=labels, right=True, include_lowest=True)
        age.append(df_demo['Age Group'].value_counts())
        political.append(df_demo['Political'].value_counts())
        religion.append(df_demo['Religion'].value_counts())

    print(religion)
    """
    # Gender
    print("----Gender----")
    print(df_demo['Sex'].value_counts())
    print(df_demo['Sex'].value_counts() / sum(df_demo['Sex'].value_counts()))

    print("----Race----")
    print(df_demo['Ethnicity simplified'].value_counts() / sum(df_demo['Ethnicity simplified'].value_counts()))

    print("----Level of Education----")
    print(df_demo['Education'].value_counts() / sum(df_demo['Education'].value_counts()))
    print(df_demo['Education'].value_counts())

    print("----Age----")
    bins = [18, 19, 29, 39, 49, 59, 69, 79, 89]
    labels = ['18-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89']
    df_demo['Age'] = df_demo['Age'].astype(int)
    df_demo['Age Group'] = pd.cut(df_demo['Age'], bins=bins, labels=labels, right=True, include_lowest=True)

    print(df_demo['Age Group'].value_counts() / sum(df_demo['Age Group'].value_counts()))
    print(df_demo['Age Group'].value_counts())

    print("----Political Orientation----")
    print(df_demo['Political'].value_counts() / sum(df_demo['Political'].value_counts()))
    print(df_demo['Political'].value_counts())

    print("----Religion----")
    religion_none = len(df_demo['Political']) - len(df_demo['Religion'])
    print(df_demo['Religion'].value_counts(dropna=False) / sum(df_demo['Religion'].value_counts(dropna=False)))
    print(df_demo['Religion'].value_counts(dropna=False))
    """
    num_annotators = len(df_demo)
    data = [
        num_annotators, '51.85', '48.15', '--', '14.81', '--', '--', '85.19', '--', 
        '--', '14.81', '33.33', '29.63', '18.52', '3.70', '--', '40.74', 
        '18.51', '18.51', '14.81', '7.4', '--', '--', '25.93', '40.74', 
        '7.41', '22.22', '3.70', '29.62', '51.85', '--', '3.70', '--', '--', 
        '7.40', '--'
    ]

    fill_latex_table(num_annotators, sex, race, education, age, political, religion)
