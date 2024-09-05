import argparse
import pandas as pd
from collections import Counter
from itertools import combinations
import numpy as np
from scipy.stats import chi2_contingency
from utils import process_language_data_descriptive


def fill_latex_table(num_annotators, sex, race, education, age, political, religion):

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
        \textbf{{Ethnicity (Simplified)}} (\%) &  &  &  &  &  \\
        \quad Asian & {asian[0]} & {asian[1]} & {asian[2]} & {asian[3]} & {asian[4]} \\ 
        \quad Black & {black[0]} & {black[1]} & {black[2]} & {black[3]} & {black[4]} \\ 
        \quad White & {white[0]} & {white[1]} & {white[2]} & {white[3]} & {white[4]} \\ 
        \quad Mixed & {mixed[0]} & {mixed[1]} & {mixed[2]} & {mixed[3]} & {mixed[4]} \\ 
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
        \quad Conservative & {conservative[0]} & {conservative[1]} & {conservative[2]} & {conservative[3]} & {conservative[4]} \\
        \quad Other & {political_other[0]} & {political_other[1]} & {political_other[2]} & {political_other[3]} & {political_other[4]} \\ \midrule
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

    TOTAL = num_annotators

    def extract_column(data, column, default="--"):
        rounded_values = [round((df[column] / TOTAL[index]) * 100, 2) if column in df and df[column]
                          != 0 else default for index, df in enumerate(data)]

        return rounded_values
    # def extract_column(data, column, default="--"):
    #    return [round(df[column], 2) if column in df and df[column] != 0 else default for index, df in enumerate(data)]

    # Sex
    male = extract_column(sex, "Male")
    female = extract_column(sex, "Female")

    # Race
    asian = extract_column(race, "Asian")
    black = extract_column(race, "Black")
    mixed = extract_column(race, "Mixed")
    white = extract_column(race, "White")
    other = extract_column(race, "Other")

    # Education
    below_high_school = extract_column(education, "Below High School")
    high_school = extract_column(education, "High School")
    college = extract_column(education, "College")
    bachelor = extract_column(education, "Bachelor")
    masters = extract_column(education, "Master’s Degree")
    doctorate = extract_column(education, "Doctorate")

    # Age Group
    age_18_19 = extract_column(age, "18-19")
    age_20_29 = extract_column(age, "20-29")
    age_30_39 = extract_column(age, "30-39")
    age_40_49 = extract_column(age, "40-49")
    age_50_59 = extract_column(age, "50-59")
    age_60_69 = extract_column(age, "60-69")
    age_70_79 = extract_column(age, "70-79")
    age_80_89 = extract_column(age, "80-89")

    # Political Party
    liberal_progressive = extract_column(political, "Liberal/Progressive")
    moderate_liberal = extract_column(political, "Moderate Liberal")
    independent = extract_column(political, "Independent")
    moderate_conservative = extract_column(political, "Moderate Conservative")
    conservative = extract_column(political, "Conservative")
    political_other = extract_column(political, "Other")

    # Religion
    none_religion = extract_column(religion, "nan")
    christian = extract_column(religion, "Christian")
    buddhism = extract_column(religion, "Buddhism")
    islam = extract_column(religion, "Islam")
    judaism = extract_column(religion, "Judaism")
    hinduism = extract_column(religion, "Hinduism")
    irreligion = extract_column(religion, "Irreligion")
    other_religion = extract_column(religion, "Other")

    # Filling out the template
    filled_template = template.format(
        annotators=num_annotators,
        male=male,
        female=female,
        asian=asian,
        black=black,
        white=white,
        mixed=mixed,
        other_race=other,
        below_high_school=below_high_school,
        high_school=high_school,
        college=college,
        bachelor=bachelor,
        masters=masters,
        doctorate=doctorate,
        age_18_19=age_18_19,
        age_20_29=age_20_29,
        age_30_39=age_30_39,
        age_40_49=age_40_49,
        age_50_59=age_50_59,
        age_60_69=age_60_69,
        age_70_79=age_70_79,
        age_80_89=age_80_89,
        liberal_progressive=liberal_progressive,
        moderate_liberal=moderate_liberal,
        independent=independent,
        moderate_conservative=moderate_conservative,
        conservative=conservative,
        political_other=political_other,
        none_religion=none_religion,
        christian=christian,
        buddhism=buddhism,
        islam=islam,
        judaism=judaism,
        hinduism=hinduism,
        irreligion=irreligion,
        other_religion=other_religion,
    )

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--annotation', '-a', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/prolific_annotations/hatespeech_main/')
    parser.add_argument('--demo', '-d', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/prolific_annotations/annotator_demo')

    args = parser.parse_args()

    num_annotators, sex, race, education, age, political, religion = process_language_data_descriptive(
        args.annotation, args.demo)

    table = fill_latex_table(num_annotators, sex, race,
                             education, age, political, religion)

    print(table)
