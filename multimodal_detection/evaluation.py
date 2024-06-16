import pandas as pd
import argparse
import os

CUTOFF = None


def extract_binary_pred(df):
    binary = df["prediction"].str.split("assistant\n\n").str[-1].str[:5].str.strip().str.lower()
    # Ensure all values are strings and apply the lambda function
    binary = binary.astype(str).apply(lambda x: 1 if "yes" in x else 0)
    df["binary"] = binary
    return df


def multimodal_examples(text_df, annotation1_df, annotation2_df, multimodal_folder):
    index = (text_df["binary"] == 0) & (annotation1_df["binary"] == 1) & (annotation2_df["binary"] == 1)
    index_1 = (text_df["binary"] == 0) & (annotation1_df["binary"] == 1)
    index_2 = (text_df["binary"] == 0) & (annotation2_df["binary"] == 1)
    index_or = index_1 | index_2

    text_df["multimodal_detection"] = index
    text_df["multimodal_detection_any"] = index_or
    text_df["multimodal_detection_1"] = index_1
    text_df["multimodal_detection_2"] = index_2
    text_df["processed_template"] = text_df["template"].apply(
        lambda x: x.split("_")[0].replace("-", " ").lower())
    annotation1_df["processed_template"] = annotation1_df["template"].apply(
        lambda x: x.split("_")[0].replace("-", " ").lower())
    annotation2_df["processed_template"] = annotation2_df["template"].apply(
        lambda x: x.split("_")[0].replace("-", " ").lower())

    result_df = pd.concat([
        text_df.groupby("processed_template")["multimodal_detection"].sum(),
        text_df.groupby("processed_template")["multimodal_detection_any"].sum(),
        text_df.groupby("processed_template")["binary"].sum(),
        text_df.groupby("processed_template")["multimodal_detection_1"].sum(),
        text_df.groupby("processed_template")["multimodal_detection_2"].sum(),
        text_df.groupby("processed_template").size()],
        axis=1,
        keys=["total_1s",
              "total_1s_any",
              "total_text_toxic",
              "total_multimodal1",
              "total_multimodal2",
              "total_samples"]
    )

    if not os.path.exists(multimodal_folder):
        os.makedirs(multimodal_folder)
    result_df.to_csv(os.path.join(multimodal_folder, 'summary.csv'))

    new_df = pd.DataFrame([], columns=["instance_id", "processed_template",
                                       "caption",
                                       "difference", "difference_any"])

    """
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
    """

    new_df["instance_id"] = text_df["instance_id"]
    new_df["caption"] = text_df["caption"]
    new_df["processed_template"] = text_df["processed_template"]
    #new_df["pred_text"] = text_df["prediction"]
    #new_df["pred_multimodal"] = annotation1["prediction"]
    new_df["difference"] = index
    new_df["difference_any"] = index_or

    # Save as excel file
    new_df.to_excel(os.path.join(multimodal_folder,
                    'results.xlsx'), index=False)


    # Extract Final Dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--annotation1', '-a1', type=str,
                        default='/lustre/project/ki-topml/minbui/projects/MultiModalHatespeech/dataset/generated_memes/multimodal_detection/duc/multimodal_detection/multimodal_detection.csv')
    parser.add_argument('--annotation2', '-a2', type=str,
                        default='/lustre/project/ki-topml/minbui/projects/MultiModalHatespeech/dataset/generated_memes/multimodal_detection/timm/multimodal_detection/multimodal_detection.csv')
    parser.add_argument('--text_file', '-t', type=str,
                        default='/lustre/project/ki-topml/minbui/projects/MultiModalHatespeech/dataset/generated_memes/multimodal_detection/text/multimodal_detection/text_detection.csv')
    parser.add_argument('--final_folder', '-f', type=str,
                        default='/lustre/project/ki-topml/minbui/projects/MultiModalHatespeech/dataset/generated_memes/final')

    args = parser.parse_args()

    annotation1 = args.annotation1
    annotation2 = args.annotation2
    text_file = args.text_file
    final_folder = args.final_folder

    text_df = pd.read_csv(text_file)
    annotation1_df = pd.read_csv(annotation1)
    annotation2_df = pd.read_csv(annotation2)

    #multimodal_df["template"] = multimodal_df["template"].str.split("_").str.get(0)
    #templates = multimodal_df["template"].unique()

    """
    for template in templates:
        subset_df = multimodal_df[multimodal_df["template"] == template]

        answer = subset_df["prediction"].str.split("assistant\n\n").str.get(-1).str.lower()
        answer = answer.apply(lambda x: 1 if "yes" in x[:5] else 0)
        count = answer.sum()
        print(f"\n------{template}-----")
        print(f"{count}/{len(subset_df)}")
    subset_df = multimodal_df[multimodal_df["template"] == "Germany Pls"]
    for index, row in subset_df.iterrows():
        print(f"\n-----{index}----")
        print(row["original"])
        print(row["prediction"].split("assistant\n\n")[-1])
    """


    if CUTOFF:
        text_df = text_df[:CUTOFF]
        multimodal_df = multimodal_df[:CUTOFF]

    text_df = extract_binary_pred(text_df)
    annotation1_df = extract_binary_pred(annotation1_df)
    annotation2_df = extract_binary_pred(annotation2_df)
    multimodal_examples(text_df, annotation1_df, annotation2_df, final_folder)
