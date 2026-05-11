"""
This step is only necessary if you are downloading data from online repositories like Zenodo. If you are using your own images, you can skip this step and move on to 1_initial_cleanup.py.

Download zip file from a specified URL, extract its contents into designated folder, and flatten the directory structure by moving all files directly into the target folder.
After extraction, delete the original zip file.
"""

import os
import shutil
import zipfile
import logging
from urllib import request
from contextlib import closing

# Setup basic logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_and_extract_zip(filename, url, output_folder):
    """
    Downloads a zip file from a URL, extracts it into the specified folder (without subfolders),
    and deletes the zip file.

    Parameters:
    - filename (str): The name to save the downloaded zip file as
    - url (str): The URL to download the zip file from
    - output_folder (str): The folder where the zip will be saved and data will be extracted into
    """
    # create the output folder where the zip file will be saved
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    zip_path = os.path.join(output_folder, filename)

    try:
        with closing(request.urlopen(url)) as r:
            with open(zip_path, 'wb') as f:
                shutil.copyfileobj(r, f)
        logger.info(f'Downloaded {filename}')
    except Exception as e:
        logger.error(f'Download failed for {filename}: {e}')
        return

    # define the folder where the ZIP file will be extracted (TEM_Raw-images)
    extract_folder = os.path.join(output_folder, 'TEM_Raw-images')

    # create folder for extracted data (directly in output_folder)
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # extract files to a temporary folder
            zip_ref.extractall(extract_folder)
        logger.info(f'Extracted {filename} to {extract_folder}')
    except Exception as e:
        logger.error(f'Extraction failed for {filename}: {e}')
        return

    # flatten the extracted contents by moving files directly into `TEM_Raw-images`
    for item in os.listdir(extract_folder):
        item_path = os.path.join(extract_folder, item)
        
        # if it's a directory, move its contents into the target folder
        if os.path.isdir(item_path):
            for sub_item in os.listdir(item_path):
                shutil.move(os.path.join(item_path, sub_item), extract_folder)

            # after moving, remove the now-empty directory
            os.rmdir(item_path)
        else:
            # otherwise, just move the file
            shutil.move(item_path, extract_folder)

    # delete the zip file after extraction
    try:
        os.remove(zip_path)
        logger.info(f'Deleted zip file: {filename}')
    except Exception as e:
        logger.warning(f'Could not delete zip file {filename}: {e}')


if __name__ == "__main__":
    url = 'https://zenodo.org/record/15253472/files/raw_data.zip?download=1'
    filename = 'raw_data.zip'
    output_folder = '.'  # this will save the zip file and extracted data in the current directory

    download_and_extract_zip(filename=filename, url=url, output_folder=output_folder)