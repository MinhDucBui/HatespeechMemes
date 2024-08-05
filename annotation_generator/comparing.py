import argparse
import pandas as pd
from langdetect import detect
import krippendorff
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--annotation1', '-a1', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/hatespeech_v2/de_duc.xlsx')
    parser.add_argument('--annotation2', '-a2', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/hatespeech_v2/de_timm.xlsx')
    parser.add_argument('--comparison', '-c', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/hatespeech_v2/de_comparison.xlsx')

    args = parser.parse_args()

    file1 = args.annotation1
    file2 = args.annotation2
    comparison_file = args.comparison
    df_annotation1 = pd.read_excel(file1)
    df_annotation2 = pd.read_excel(file2)
    print(df_annotation1["Hate Speech (=1) or Non-Hate Speech (=0)"])
    reliability_data = [list(df_annotation1["Hate Speech (=1) or Non-Hate Speech (=0)"].astype(int)),
                        list(df_annotation2["Hate Speech (=1) or Non-Hate Speech (=0)"])]
    alpha = krippendorff.alpha(
        reliability_data=reliability_data, level_of_measurement="nominal")
    print("Krippendorff's alpha: ", alpha)
    diff = df_annotation1["Hate Speech (=1) or Non-Hate Speech (=0)"].astype(
        int) == df_annotation2["Hate Speech (=1) or Non-Hate Speech (=0)"]
    print("Difference: {}/{}".format(sum(diff), len(df_annotation1)))

    discrepancies = df_annotation1["Hate Speech (=1) or Non-Hate Speech (=0)"] != df_annotation2["Hate Speech (=1) or Non-Hate Speech (=0)"]
    df_annotation1["Timm_Hate"] = df_annotation2["Hate Speech (=1) or Non-Hate Speech (=0)"]
    df_annotation1["Timm"] = df_annotation2["Hate Speech (=1) or Non-Hate Speech (=0)"]

    df_comparison = pd.read_excel(comparison_file)
    # repeating_list = [i for i in range(0, 17 + 1) for _ in range(10)]
    print(df_comparison.keys())
    # df_comparison["Group_ID"] = repeating_list + repeating_list
    value_counts = df_comparison['Why'].value_counts()
    print(value_counts)
    # grouped_counts = df_comparison.groupby('Group_ID')['Why'].value_counts()
    # print(grouped_counts)
