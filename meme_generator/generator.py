import argparse
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import pandas as pd
import os
from tqdm import tqdm
from langdetect import detect


LANGUAGES = ["en", "de"]


def make_meme(topString, bottomString, filename, output_file, font_path="/Library/Fonts/Impact.ttf", font_size=20):
    # Open the image and get its size
    img = Image.open(filename)
    image_size = img.size
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)

    # Function to get text size using textbbox
    def get_text_size(text, font):
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        return text_width, text_height

    # Function to split text into multiple lines if it exceeds the image width
    def split_text(text, font, max_width):
        words = text.split()
        lines = []
        current_line = []

        for word in words:
            current_line.append(word)
            if get_text_size(' '.join(current_line), font)[0] > max_width:
                current_line.pop()
                lines.append(' '.join(current_line))
                current_line = [word]

        lines.append(' '.join(current_line))
        return lines

    # Split top and bottom strings if needed
    max_width = image_size[0] - 20  # 10 pixels padding on each side
    top_lines = split_text(topString, font, max_width)
    bottom_lines = split_text(bottomString, font, max_width)

    # Calculate positions for top and bottom text
    top_text_position_y = 10  # Padding from the top
    bottom_text_height = sum([get_text_size(line, font)[1] for line in bottom_lines]) + 10 * (len(bottom_lines) - 1)
    bottom_text_position_y = image_size[1] - bottom_text_height - 10  # Padding from the bottom

    # Draw outlines and text for top lines
    outline_range = 2
    for i, line in enumerate(top_lines):
        text_width, text_height = get_text_size(line, font)
        text_position_x = (image_size[0] - text_width) / 2
        text_position_y = top_text_position_y + i * (text_height + 10)

        for x in range(-outline_range, outline_range + 1):
            for y in range(-outline_range, outline_range + 1):
                draw.text((text_position_x + x, text_position_y + y), line, (0, 0, 0), font=font)
        draw.text((text_position_x, text_position_y), line, (255, 255, 255), font=font)

    # Draw outlines and text for bottom lines
    for i, line in enumerate(bottom_lines):
        text_width, text_height = get_text_size(line, font)
        text_position_x = (image_size[0] - text_width) / 2
        text_position_y = bottom_text_position_y + i * (text_height + 10)

        for x in range(-outline_range, outline_range + 1):
            for y in range(-outline_range, outline_range + 1):
                draw.text((text_position_x + x, text_position_y + y), line, (0, 0, 0), font=font)
        draw.text((text_position_x, text_position_y), line, (255, 255, 255), font=font)

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
