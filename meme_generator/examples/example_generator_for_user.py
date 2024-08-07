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


LANGUAGES = ["en", "de"]


FONT_MAPPING = {
    "de": "Impact.ttf",
    "en": "Impact.ttf",
    "es": "Impact.ttf",
    "zh": "/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/Arial-Unicode-Bold.ttf",
    "hi": "/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/Arial-Unicode-Bold.ttf",
}

FONT_HEIGHT = {
    "de": 45,
    "en": 45,
    "es": 45,
    "zh": 45,
    "hi": 45,
}


def get_examples(language):
    if language == "de":
        examples = {
            "1134290.jpg": ["SPRICHST IHN AUF FRANZÖSISCH AN", "ANTWORTET DIR IN EIN UNVERSTÄNDLICHEM ENGLISCH"],
            "699717.jpg": ["SCHEIß AUF", "STEVE JOBS"],
            "2061647.jpg": ["NIE STEUERN ZAHLEN", "UND IMMER BETRUNKEN"],  # Native
            "1436.jpg": ["JIAN MIN SAGT ER IST EIN GUTER FAHRER", "GUTER WITZ"],  # Asian Male
            "332838_a.jpg": ["DAS IST DEIN NEUER NACHBAR", "DU MUSS JETZT VORSICHTIGER SEIN"],
            "332838.jpg": ["SCHEIßE JA", "IST ES FREITAG"],
        }
    if language == "en":
        examples = {
            "1134290.jpg": ["TALK TO HIM IN FRENCH", "ANSWERS YOU WITH AN INCOMPREHENSIBLE ENGLISH"],
            "2061647.jpg": ["NEVER PAYING TAXES", "AND ALWAYS DRUNK"],
            "1436.jpg": ["JIAN MIN SAYS HES A GOOD DRIVER", "GOOD JOKE"],  # Asian Male
            "332838_a.jpg": ["THIS IS YOUR NEW NEIGHBOR", "YOU HAVE TO BE MORE CAREFUL NOW"], # Non Hate Muslim Crowd
            "332838.jpg": ["FUCK YEAH", "ITS FRIDAY"],  # Non Muslim Man
        }

    return examples


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


def make_meme(topString, bottomString, filename, output_file, language):

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')
    parser.add_argument('--image_folder', '-i', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/examples/input/')
    parser.add_argument('--output_path', '-o', type=str,
                        default='/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/examples/output/user_')
    args = parser.parse_args()

    for language in LANGUAGES:
        image_folder = args.image_folder
        output_path = args.output_path
        output_path = output_path + language
        font = FONT_MAPPING[language]

        # filename = image_folder.split("/")[-1]
        # output_filename = os.path.join(output_path, filename)

        examples = get_examples(language)

        for key, value in examples.items():
            output_filename = os.path.join(output_path, key)
            image_file = os.path.join(image_folder, key)
            top = value[0]
            bottom = value[1]
            make_meme(top, bottom, image_file,
                      output_filename, language=language)
