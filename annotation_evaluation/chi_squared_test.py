import argparse
import pandas as pd
from itertools import combinations
import numpy as np
from scipy.stats import chi2_contingency
from utils import process_language_data


def get_latex_table(df_counts, df_percent):
    COUNTRY_LIST = ["US", "DE", "MX", "IN", "CN"]
    print("\n\n ------Latex------")
    total_counts = df_counts.sum(axis=1)
    # Constructing the LaTeX table
    latex_table = "\\begin{table*}\n\\centering\n\\small\n\\setlength{\\tabcolsep}{5pt}\n"
    latex_table += "\\begin{tabular}{l|ccccc}\n"
    latex_table += "\\toprule\n"
    latex_table += "& USA & Germany & Mexico & India & China  \\\\\n"
    latex_table += "\\midrule\n"

    # Hatespeech row
    hatespeech_row = "Hatespeech & "
    hatespeech_row += " & ".join([
        f"{int(df_counts[1][country])} ({int(df_percent[1][country] * 100)}\\%)"
        for country in COUNTRY_LIST
    ])
    hatespeech_row += " \\\\\n"

    # Non-Hatespeech row
    non_hatespeech_row = "Non-Hatespeech & "
    non_hatespeech_row += " & ".join([
        f"{int(df_counts[0][country])} ({int(df_percent[0][country] * 100)}\\%)"
        for country in COUNTRY_LIST
    ])
    non_hatespeech_row += " \\\\\n"

    # Total row
    total_row = "Total & "
    total_row += " & ".join(
        [f"{int(total_counts[country])}" for country in COUNTRY_LIST])
    total_row += " \\\\\n"

    # Combine rows
    latex_table += hatespeech_row
    latex_table += non_hatespeech_row
    latex_table += "\\midrule\n"
    latex_table += total_row

    # Closing the table
    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\end{table*}"

    # Print the LaTeX table
    print(latex_table)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--annotation', '-a', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/prolific_annotations/hatespeech_main/')

    args = parser.parse_args()
    merged_df = process_language_data(args.annotation)
    merged_df = merged_df.reset_index()

    # Melt the DataFrame to transform it from wide to long format
    merged_df = pd.melt(merged_df, id_vars=[
                        'ID'], var_name='Nationality', value_name='hatespeech')

    def create_contingency_table(df, attribute):
        return pd.crosstab(df[attribute], df['hatespeech'], margins=False)

    contingency_country = create_contingency_table(merged_df, 'Nationality')
    # print(contingency_country)
    # diff = -5
    # count_ = "US"
    # contingency_country.loc[count_][0] = contingency_country.loc[count_][0] - diff
    # contingency_country.loc[count_][1] = contingency_country.loc[count_][1] + diff
    row_sums = contingency_country.sum(axis=1)
    normalized_df = contingency_country.div(row_sums, axis=0)

    chi2, p_value_country, dof, expected = perform_chi_squared_test(
        contingency_country)

    alpha = 0.05

    results = {
        'country': p_value_country,
    }

    for attribute, p_value in results.items():
        if p_value < alpha:
            print(
                f'There are significant disparities in annotations across {attribute} (p = {p_value})')
        else:
            print(
                f'There are no significant disparities in annotations across {attribute} (p = {p_value})')

    get_latex_table(contingency_country, normalized_df)
