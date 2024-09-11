from openai import OpenAI
import base64
import argparse
from utils import process_language_data
import pandas as pd
import os
from tqdm import tqdm  # Import tqdm

LANGUAGES = ["en"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--annotation', '-a', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/prolific_annotations/hatespeech_main/')
    parser.add_argument('--results', '-r', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/responses_')

    args = parser.parse_args()
    df = process_language_data(args.annotation)
    # df = df[df["MX_unanimously"] == True]

    file_path = args.results + LANGUAGES[0] + ".csv"
    df_results = pd.read_csv(file_path)
    df_results["response_" + LANGUAGES[0]] = df_results["response"]
    mapping = {'Yes.': 1, 'No.': 0}
    for language in LANGUAGES[1:]:
        file_path = args.results + language + ".csv"
        df_results_part = pd.read_csv(file_path)
        df_results = pd.merge(df_results, df_results_part, on="ID", how="inner", suffixes=('', "_" + language))
        df_results['response_' + language] = df_results['response_' + language].map(mapping)
    df_results['response_' + LANGUAGES[0]] = df_results['response_' + LANGUAGES[0]].map(mapping)
    df_results = df_results.dropna()
    print(df_results)
    df = df.reset_index()
    df["ID"] = df["ID"].astype(str)
    df_results["ID"] = df_results["ID"].astype(str)
    df_all = pd.merge(df_results, df, on="ID", how="inner")
    print(df_all)
    COUNTRIES = ["US", "DE", "MX", "CN", "IN"]
    for language in LANGUAGES:
        print("---")
        for country in COUNTRIES:
            print("{} with {}: {}".format(language, country, round(sum(df_all['response_' + language] == df_all[country]) / len(df_all) * 100, 2)))

    print("---")

    df_subset = df_all
    for language in LANGUAGES[1:]:
        print("------" + language)
        df_subset_ = df_subset[df_subset["response_en"] == df_subset['response_' + language]]

        print("Overlapp of {}".format(len(df_subset_) / len(df_all) * 100))
