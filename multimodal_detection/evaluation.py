import pandas as pd
import argparse
import os

CUTOFF = None


def extract_binary_pred(df):
    binary = df["prediction"].str.split(
        "Answer: ").str[-1].str.strip().str.lower()
    # Ensure all values are strings and apply the lambda function
    binary = binary.astype(str).apply(lambda x: 1 if "yes" in x else 0)
    df["binary"] = binary
    return df


def multimodal_examples(text_df, multimodal_df, multimodal_folder):
    index = (text_df["binary"] == 0) & (multimodal_df["binary"] == 1)
    text_df["multimodal_detection"] = index
    text_df["processed_template"] = text_df["template"].apply(
        lambda x: x.split("_")[0].replace("-", " ").lower())
    multimodal_df["processed_template"] = multimodal_df["template"].apply(
        lambda x: x.split("_")[0].replace("-", " ").lower())

    result_df = pd.concat([
        text_df.groupby("processed_template")["multimodal_detection"].sum(),
        text_df.groupby("processed_template")["binary"].sum(),
        multimodal_df.groupby("processed_template")["binary"].sum(),
        text_df.groupby("processed_template").size(),
        multimodal_df.groupby("processed_template").size()],
        axis=1,
        keys=["total_1s",
              "total_text_toxic",
              "total_multimodal_toxic",
              "total_samples_text",
              "total_samples_multimodal"]
    )

    result_df.to_csv(os.path.join(multimodal_folder, 'summary.csv'))

    new_df = pd.DataFrame([], columns=["instance_id", "processed_template",
                                       "caption",
                                       "pred_text", "pred_multimodal",
                                       "difference"])
    NUM_EXAMPLE = 0
    for template in list(set(list(text_df["processed_template"]))):
        filtered_rows = text_df[(
            text_df["processed_template"] == template) & index].copy().iloc[:NUM_EXAMPLE]
        examples = list(filtered_rows["caption"])
        preds_text = list(filtered_rows["prediction"])
        filtered_rows = multimodal_df[(
            multimodal_df["processed_template"] == template) & index].iloc[:NUM_EXAMPLE]
        preds_multimodal = list(filtered_rows["prediction"])
        if NUM_EXAMPLE > 0:
            print(f"\n\n------{template}------")
            for example, pred_text, pred_multimodal in zip(examples, preds_text, preds_multimodal):
                print("\n--example--")
                print(example)
                print(pred_text)
                print(pred_multimodal)

    new_df["instance_id"] = text_df["instance_id"]
    new_df["caption"] = text_df["caption"]
    new_df["processed_template"] = text_df["processed_template"]
    new_df["pred_text"] = text_df["prediction"]
    new_df["pred_multimodal"] = multimodal_df["prediction"]
    new_df["difference"] = index

    # Save as excel file
    new_df.to_excel(os.path.join(multimodal_folder,
                    'results.xlsx'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--multimodal_folder', '-m', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/generated_memes/multimodal_detection')

    args = parser.parse_args()

    multimodal_folder = args.multimodal_folder
    text_df = pd.read_csv(os.path.join(
        multimodal_folder, "text_detection.csv"))
    multimodal_df = pd.read_csv(os.path.join(
        multimodal_folder, "multimodal_detection.csv"))

    if CUTOFF:
        text_df = text_df[:CUTOFF]
        multimodal_df = multimodal_df[:CUTOFF]

    text_df = extract_binary_pred(text_df)
    multimodal_df = extract_binary_pred(multimodal_df)
    multimodal_examples(text_df, multimodal_df, multimodal_folder)