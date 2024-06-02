import argparse
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import pandas as pd
import os
from tqdm import tqdm
from langdetect import detect


LANGUAGES = ["en", "de"]


def make_meme(topString, bottomString, filename, output_file):
    img = Image.open(filename)
    imageSize = img.size

    # Find the biggest font size that works
    fontSize = int(imageSize[1] / 5)
    font = ImageFont.truetype("/Library/Fonts/Impact.ttf", fontSize)
    draw = ImageDraw.Draw(img)

    # Function to get text size using textbbox
    def get_text_size(text, font):
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        return (text_width, text_height)

    topTextSize = get_text_size(topString, font)
    bottomTextSize = get_text_size(bottomString, font)

    while topTextSize[0] > imageSize[0] - 20 or bottomTextSize[0] > imageSize[0] - 20:
        fontSize -= 1
        font = ImageFont.truetype("/Library/Fonts/Impact.ttf", fontSize)
        topTextSize = get_text_size(topString, font)
        bottomTextSize = get_text_size(bottomString, font)

    # Find top centered position for top text
    topTextPositionX = (imageSize[0] / 2) - (topTextSize[0] / 2)
    topTextPositionY = 8  # Padding from the top
    topTextPosition = (topTextPositionX, topTextPositionY)

    # Find bottom centered position for bottom text
    bottomTextPositionX = (imageSize[0] / 2) - (bottomTextSize[0] / 2)
    bottomTextPositionY = imageSize[1] - bottomTextSize[1] - 15  # Padding from the bottom
    bottomTextPosition = (bottomTextPositionX, bottomTextPositionY)

    # Draw outlines
    outlineRange = int(fontSize / 15)
    for x in range(-outlineRange, outlineRange + 1):
        for y in range(-outlineRange, outlineRange + 1):
            draw.text(
                (topTextPosition[0] + x, topTextPosition[1] + y), topString, (0, 0, 0), font=font)
            draw.text(
                (bottomTextPosition[0] + x, bottomTextPosition[1] + y), bottomString, (0, 0, 0), font=font)

    # Draw the text
    draw.text(topTextPosition, topString, (255, 255, 255), font=font)
    draw.text(bottomTextPosition, bottomString, (255, 255, 255), font=font)

    img.save(output_file, 'JPEG')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--memes', '-m', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/memes')
    parser.add_argument('--generate_folder', '-g', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/generated_memes')
    # parser.add_argument('--save-dir', '-d', required=True, type=str,
    #                    help='directory where the dataset should be stored')
    args = parser.parse_args()

    folder_path = args.memes
    output_folder = args.generate_folder
    headers = ['label', 'template', 'image']  # Replace with your actual column names

    # Load the text file into a DataFrame
    df_template = pd.read_csv(os.path.join(folder_path, "templates.txt"), sep='\t', names=headers)
    df_template = df_template.drop_duplicates()

    for language in LANGUAGES:
        headers = ['template', 'instance_id', 'caption_translated', 'caption_original']
        # Load the text file into a DataFrame
        df = pd.read_csv(os.path.join(output_folder, "caption_translation", language + ".txt"), sep='\t', names=headers)
        df = df.drop_duplicates()
        template_ids = []

        for index, row in tqdm(df.iterrows()):
            # Create Path
            template_name = row["template"].split("/")[-1]
            output_path = os.path.join(output_folder, "images", template_name, language)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            output_path = os.path.join(output_path, str(row["instance_id"]) + ".jpg")

            # Get Bottom & Top Strings
            top = row["caption_translated"].split("<sep>")[0].strip()
            bottom = row["caption_translated"].split("<sep>")[1].strip()

            if top == "<emp>":
                top = ""
            if bottom == "<emp>":
                bottom = ""

            # Get Image Path
            template_name = row["template"]
            if not df_template.empty:
                # For existing Deephumor dataset
                image_file = df_template[df_template["label"] == template_name]["image"].iloc[0].split("/")[-1]
                image_file = os.path.join(folder_path, "images", image_file)
            else:
                # For own crawled dataset
                image_file = os.path.join(folder_path, "images",
                                          template_name + ".jpg")
            make_meme(top, bottom, image_file, output_path)
            template_ids.append(str(row["instance_id"]))
