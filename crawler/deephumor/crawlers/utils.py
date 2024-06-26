import os
import shutil

import requests


def time_to_str(time):
    """Converts time in seconds into pretty-looking string."""
    return f'{int(time / 60.):3d}:{(time % 60.):05.2f}'


def load_image(name, image_url, save_dir='.'):
    """Loads image by url.

    Args:
        image_url (str): image URL
        save_dir (str): directory for saving the image

    Returns:
        str: name of the file
    """
    # Extract the file name from the URL
    file_name = name + "_variant=" + image_url.split('/')[-1]
    image_path = os.path.join(save_dir, file_name)

    # Check if the image already exists
    if os.path.exists(image_path):
        return file_name

    # Download the image if it doesn't exist
    r = requests.get(image_url, stream=True)
    if r.status_code == 200:
        with open(image_path, 'wb') as out:
            shutil.copyfileobj(r.raw, out)
        print(f"Image downloaded and saved: {image_path}")
    else:
        print(f"Failed to download image. Status code: {r.status_code}")

    return image_path
