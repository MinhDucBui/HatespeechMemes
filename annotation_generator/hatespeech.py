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
LANGUAGES = ["hi"]


FONT_MAPPING = {
    "de": "Impact.ttf",
    "en": "Impact.ttf",
    "es": "Impact.ttf",
    "zh": "/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/Arial-Unicode-Bold.ttf",
    "hi": "/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/Arial-Unicode-Bold.ttf"
}

FONT_HEIGHT = {
    "de": 45,
    "en": 45,
    "es": 45,
    "zh": 45,
    "hi": 45,
}

PADDING_BOTTOM = {
    "hi": 40
}

# Load a blank SpaCy model
NLP = spacy.blank("en")
# Define a custom tokenizer function using Jieba


def jieba_tokenizer(text):
    tokens = jieba.lcut(text)
    return Doc(NLP.vocab, words=tokens)


# Set the tokenizer to use Jieba
NLP.tokenizer = jieba_tokenizer


def resize_image(filename, new_width, new_height):
    # Open the image
    img = Image.open(filename)

    # Resize the image
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return resized_img


def make_meme(topString, bottomString, filename, output_file, worksheet, excel_index, language):

    # Font Sizes
    font_path = FONT_MAPPING[language]
    font_size = 50

    # Open the image and get its size
    img = Image.open(filename)
    new_width = 600
    new_height = 600

    img = resize_image(filename, new_width, new_height)
    image_size = img.size
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)

    # Function to get text size using textbbox
    def get_text_size(text, font):
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_height = FONT_HEIGHT[language]
        return text_width, text_height

    # Function to split text into multiple lines if it exceeds the image width
    def split_text(text, font, max_width, language):

        if language == "zh":
            doc = NLP(text)
            words = [token.text for token in doc if token.text != " "]
        else:
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
    top_lines = split_text(topString, font, max_width, language=language)
    bottom_lines = split_text(bottomString, font, max_width, language=language)

    # Calculate positions for top and bottom text
    top_text_position_y = 10  # Padding from the top
    bottom_text_height = sum([get_text_size(line, font)[1]
                             for line in bottom_lines]) + 10 * (len(bottom_lines) - 1)
    # Padding from the bottom
    if language in PADDING_BOTTOM.keys():
        bottom_text_position_y = image_size[1] - bottom_text_height - PADDING_BOTTOM[language]
    else:
        bottom_text_position_y = image_size[1] - bottom_text_height - 25

    # Draw outlines and text for top lines
    outline_range = 2
    for i, line in enumerate(top_lines):
        text_width, text_height = get_text_size(line, font)
        text_position_x = (image_size[0] - text_width) / 2
        text_position_y = top_text_position_y + i * (text_height + 10)

        for x in range(-outline_range, outline_range + 1):
            for y in range(-outline_range, outline_range + 1):
                draw.text((text_position_x + x, text_position_y + y),
                          line, (0, 0, 0), font=font)
        draw.text((text_position_x, text_position_y),
                  line, (255, 255, 255), font=font)

    # Draw outlines and text for bottom lines
    for i, line in enumerate(bottom_lines):
        text_width, text_height = get_text_size(line, font)

        text_position_x = (image_size[0] - text_width) / 2
        text_position_y = bottom_text_position_y + i * (text_height + 10)

        for x in range(-outline_range, outline_range + 1):
            for y in range(-outline_range, outline_range + 1):
                draw.text((text_position_x + x, text_position_y + y),
                          line, (0, 0, 0), font=font)
        draw.text((text_position_x, text_position_y),
                  line, (255, 255, 255), font=font)
    # Define the scaling factor (e.g., reduce by 50%)
    scale_factor = 0.4
    new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    img.save(output_file, 'JPEG')

    # Convert the dataframe to an XlsxWriter Excel object.
    # col_idx = df.columns.get_loc('ID') + 1
    # writer.sheets['Sheet1'].set_column(col_idx, col_idx, 40)
    worksheet.insert_image('C' + str(excel_index), output_file)
    # Close the Pandas Excel writer and output the Excel file.
    # das


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--memes', '-m', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/memes')
    parser.add_argument('--final_dataset', '-f', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/translation_final/done/with_non_hate')
    parser.add_argument('--generate_folder', '-g', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/generated_memes')
    parser.add_argument('--output_folder', '-o', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/hatespeech_hindi')
    parser.add_argument('--test_run', '-t',
                        # This will set the default to False and set it to True if the flag is present
                        action='store_true',
                        help='Set this flag to enable test run mode. If omitted, test run mode is disabled by default.')

    args = parser.parse_args()

    folder_path = args.memes
    generate_folder = args.generate_folder
    output_folder = args.output_folder
    final_dataset = args.final_dataset
    test_run = args.test_run

    templates_processed = []
    for language in LANGUAGES:
        # Replace with your actual column names
        headers = ['template',
                   'instance_id',
                   'caption_translated',
                   'caption_original']

        # Load the text file into a DataFrame
        df_caption = pd.read_csv(os.path.join(
            generate_folder, "caption_translation",  "en.txt"), sep='\t', names=headers)
        df_caption = df_caption.drop_duplicates()

        headers = ['template', 'instance_id',
                   'caption_translated', 'caption_original']
        # Load the text file into a DataFrame
        final_file = os.path.join(
            final_dataset, language + "_translation.xlsx")
        df_annotation = pd.read_excel(final_file)
        df_annotation = df_annotation.dropna(subset=['ID'])
        # df_annotation = df_annotation[df_annotation["ID"] == -329162.0]

        # Remove zero-width space character (\u200b)
        df_annotation['Translation'] = df_annotation['Translation'].apply(
            lambda x: x if pd.isna(x) else x.replace('\u200b', ''))
        df_annotation['Correct Translation'] = df_annotation['Correct Translation'].apply(
            lambda x: x if pd.isna(x) else x.replace('\u200b', ''))
        # df_annotation = df_annotation[df_annotation['ID'].isin([56603606, 14106131, 11807098, 44355237])]
        # df_annotation = df_annotation[df_annotation['ID'].isin([72039904])]
        df_annotation = df_annotation.drop_duplicates()
        df_annotation = df_annotation.sample(frac=1).reset_index(drop=True)
        instances_processing = list(df_annotation["ID"])

        # Create a Pandas Excel writer using XlsxWriter as the engine.
        # Replace with your desired column names
        column_names = ['ID', 'Image', 'Caption',
                        'Hate Speech (=1) or Non-Hate Speech (=0)', 'Comments']
        df_hatespeech = pd.DataFrame(columns=column_names)
        df_hatespeech["ID"] = instances_processing

        # Create a mask for rows where 'Correct Translation' is not empty
        non_empty = pd.notna(df_annotation['Correct Translation'])
        df_annotation.loc[non_empty,
                          'Translation'] = df_annotation.loc[non_empty, 'Correct Translation']
        if "2nd: Correct Translation" in df_annotation.keys():
            non_empty = pd.notna(df_annotation['2nd: Correct Translation'])
            df_annotation.loc[non_empty, 'Translation'] = df_annotation.loc[non_empty,
                                                                            '2nd: Correct Translation']

        if "3nd: Correct Translation" in df_annotation.keys():
            non_empty = pd.notna(df_annotation['3nd: Correct Translation'])
            df_annotation.loc[non_empty, 'Translation'] = df_annotation.loc[non_empty,
                                                                            '3nd: Correct Translation']
        df_hatespeech["Caption"] = df_annotation["Translation"]
        df_hatespeech['Caption'] = df_hatespeech['Caption'].apply(
            lambda x: x if pd.isna(x) else x.replace('\u200b', ''))

        writer = pd.ExcelWriter(os.path.join(
            output_folder, language + ".xlsx"), engine='xlsxwriter')
        df_hatespeech.to_excel(writer, sheet_name='Sheet1')
        # Get the xlsxwriter workbook and worksheet objects.
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        worksheet.set_default_row(200)
        col_idx = df_hatespeech.columns.get_loc('Image') + 1
        writer.sheets['Sheet1'].set_column(col_idx, col_idx, 40)
        # col_idx = df_hatespeech.columns.get_loc('Caption') + 1
        # writer.sheets['Sheet1'].set_column(col_idx, col_idx, 40)
        col_idx = df_hatespeech.columns.get_loc('ID') + 1
        writer.sheets['Sheet1'].set_column(col_idx, col_idx, 10)
        col_idx = df_hatespeech.columns.get_loc(
            'Hate Speech (=1) or Non-Hate Speech (=0)') + 1
        writer.sheets['Sheet1'].set_column(col_idx, col_idx, 20)
        col_idx = df_hatespeech.columns.get_loc('Comments') + 1
        writer.sheets['Sheet1'].set_column(col_idx, col_idx, 40)
        for index, instance in tqdm(enumerate(instances_processing)):
            row = df_caption[df_caption["instance_id"] == instance].iloc[0]
            caption = df_hatespeech[df_hatespeech["ID"]
                                    == instance]["Caption"].iloc[0]
            # for index, row in tqdm(df_caption.iterrows()):
            if row["instance_id"] not in instances_processing:
                continue

            # Create Path
            template_name = row["template"].split("/")[-1].split("_")[0]
            template_name = template_name.replace(" ", "-")

            if test_run:
                top = "Text Text Text Text Text Text Text Text Text Text Text Text Text Text Text Text Text Text "
                bottom = "Text Text Text Text Text Text Text Text Text Text Text Text Text Text Text Text Text Text Text Text Text Text Text Text"
                if template_name in templates_processed:
                    continue
                templates_processed.append(template_name)
                # quick fix
                if "asian" not in template_name.lower():
                    continue

                output_path = os.path.join(
                    output_folder, "images", language, template_name)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                output_path = os.path.join(
                    output_path, str(row["instance_id"]) + ".jpg")
            else:

                output_path = os.path.join(
                    output_folder, "images", language, template_name)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                output_path = os.path.join(
                    output_path, str(row["instance_id"]) + ".jpg")
                # Get Bottom & Top Strings
                top = caption.split("<sep>")[0].strip()
                bottom = caption.split("<sep>")[1].strip()

                if top == "<emp>":
                    top = ""
                if bottom == "<emp>":
                    bottom = ""

            # Get Image Path
            template_name = df_caption[df_caption["instance_id"]
                                       == row["instance_id"]]["template"].iloc[0]
            template_name = template_name.replace(" ", "-")
            image_file = os.path.join(
                folder_path, "images", template_name + ".jpg")

            if not os.path.exists(image_file):
                image_file = os.path.join(
                    folder_path + "900k", "images", template_name + ".jpg")

            font = FONT_MAPPING[language]
            make_meme(top, bottom, image_file, output_path,
                      worksheet, index+2, language=language)

        writer.book.close()
