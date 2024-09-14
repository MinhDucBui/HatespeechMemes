import pandas as pd
import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
two_dirs_up = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
sys.path.append(two_dirs_up)

from annotation_evaluation.utils import process_language_data
import argparse
from tqdm import tqdm

tqdm.pandas()

REMOVE_PROMPTS = [
    "(bug) Would you consider this meme as a hate speech? Give an answer in a single alphabet only.",
    "Do you think this is a hate speech? Answer with only one single alphabet between 'a' and 'b', with no other description.",
    "Would you consider this meme as a hate speech? Give an answer in a single alphabet only.",
]

MAPPING = {
    "Answer if this meme is hate or not with a single alphabet letter among given answer choices a and b.": 0,
    "Given a meme from online, answer if it is 'Hate Speech' or 'Non-Hate Speech'. Answer with only one single alphabet between 'a' and 'b', with no other description.": 1,
    "(bug) Would you consider this meme as a hate speech? Give an answer in a single alphabet only.": 2,
    "Do you think this is a hate speech? Answer with only one single alphabet between 'a' and 'b', with no other description.": 3,
    "Would you consider this meme as a hate speech? Give an answer in a single alphabet only.": 4,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--file', '-a', type=str, default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/model_predictions/models--gpt_4o_caption/responses_en.csv')
    args = parser.parse_args()

    file_path = args.file
    df_file = pd.read_csv(file_path)

    for remove in REMOVE_PROMPTS:
        prompt_id = MAPPING[remove]
        prompt_id = prompt_id * 2
        df_file = df_file[df_file["prompt"] != prompt_id]
        df_file = df_file[df_file["prompt"] != prompt_id+1]

    file_path = file_path.split(".csv")[0] + "_cut.csv"
    df_file.to_csv(file_path, index=False)