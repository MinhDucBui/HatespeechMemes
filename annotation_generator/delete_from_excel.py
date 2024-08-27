import argparse
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import pandas as pd
import os
from tqdm import tqdm
import spacy
import jieba
from spacy.tokens import Doc


LANGUAGES = ["en"]
LANGUAGES = ["en", "de", "hi", "es", "zh"]
LANGUAGES = ["es"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--file', '-f', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/translation_final/done')
    parser.add_argument('--checking_file', '-c', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/checking_wordplay_final')
    parser.add_argument('--non_hate_folder', '-n', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/translation_nonhate/done')

    args = parser.parse_args()

    file_path = args.file
    checking_path = args.checking_file
    non_hate_folder = args.non_hate_folder
    headers = ["index", "ID", "Template Name",
               "Original (English)", "Translation", "Correct (=1) or False (=0)", "Correct Translation"]
    df = pd.read_excel(os.path.join(
        file_path, "en_translation.xlsx"), names=headers)

    orig_keys = list(df.keys())
    print(orig_keys)
    df_random = pd.read_excel(os.path.join(checking_path, "random.xlsx"))
    df_random = df_random[["instance_id"]]
    df_random["random"] = 1
    df_random = df_random.drop_duplicates()

    df_multimodal = pd.read_excel(
        os.path.join(checking_path, "multimodal.xlsx"))
    df_multimodal = df_multimodal[["instance_id"]]
    df_multimodal["multimodal"] = 1
    df_multimodal = df_multimodal.drop_duplicates()

    merged_df = pd.merge(df, df_random, left_on='ID',
                         right_on='instance_id', how='left')
    merged_df = merged_df[orig_keys + ["random"]]

    merged_df = pd.merge(merged_df, df_multimodal, left_on='ID',
                         right_on='instance_id', how='left')
    merged_df = merged_df[orig_keys + ["random", "multimodal"]]

    merged_df = merged_df.sort_values(by='multimodal')
    merged_df = merged_df.drop_duplicates()

    # Identify the last row of each template name
    last_indices = merged_df[merged_df['Template Name'].notna()].groupby(
        ['Template Name', "random"]).apply(lambda x: x.index[-1]).tolist()
    merged_df = merged_df.drop(index=last_indices)
    merged_df = merged_df.sort_values(by='multimodal')

    last_indices = merged_df[merged_df['Template Name'].notna()].groupby(
        ['Template Name', "multimodal"]).apply(lambda x: x.index[-1]).tolist()
    merged_df = merged_df.drop(index=last_indices)

    ids = list(merged_df["ID"])

    output_dir = os.path.join(file_path, 'with_non_hate')
    os.makedirs(output_dir, exist_ok=True)

    for language in LANGUAGES:
        file_name_base = language + "_translation.xlsx"
        output_file = os.path.join(output_dir, file_name_base)
        df = pd.read_excel(os.path.join(file_path, file_name_base))

        df = df[df["ID"].isin(ids)]
        #print(df[df["Template Name"] == "success germany"]["ID"])

        df_nonhate = pd.read_excel(
            os.path.join(non_hate_folder, file_name_base))

        df = pd.concat([df, df_nonhate])
        df.to_excel(output_file, index=False)
