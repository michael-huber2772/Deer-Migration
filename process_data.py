import os
import shutil
import pandas as pd
import re


def process_data():
    """
    This function will create the directories data/training, data/training/animal,
    data/training/blank, and data/unlabelled and will populate these directories with
    images.
    """

    try:
        os.mkdir('data/training')
    except OSError:
        print('Creation of training directory failed')
    else:
        print('Creation of training directory successful')

    try:
        os.mkdir('data/training/animal')
    except OSError:
        print("Creation of animal directory failed")
    else:
        print('Creation of animal Directory Successful')

    try:
        os.mkdir('data/training/blank')
    except OSError:
        print("Creation of blank directory failed")
    else:
        print('Creation of blank Directory Successful')

    try:
        os.mkdir('data/unlabelled')
    except OSError:
        print("Creation of unlabelled directory failed")
    else:
        print('Creation of unlabelled Directory Successful')

    label_files = []
    image_dirs = []

    # Loop through the files and determine if they are folders containing images or data files
    for file in os.listdir('data/MuleDeetData/'):
        if '.xlsx' in file:  # If .xlsx is in file name, the file contains data
            label_files.append(file)
        else:  # If not, then it will be a directory with images
            image_dirs.append(file)

    # Create a dataframe of all labelled data
    all_data = pd.DataFrame()
    for file in label_files:  # loop through all .xlsx files
        df = pd.read_excel(os.path.join('data/MuleDeetData/', file))  # read in next file
        all_data = all_data.append(df, ignore_index=True)  # append to dataframe

    people = ['Start', 'Setup/Pickup', 'End']  # List of labels that identify people
    for i in range(len(all_data)):  # Loop through all images in datafile

        # Get the name of the folder that contains the image
        folder = re.sub('_(\d)+\.(?i)JPG$', '', all_data['Raw Name'][i])

        # Get the name and location of the image
        image = os.path.join('data/MuleDeetData', folder, all_data['Raw Name'][i])

        # Check if the image has been labelled and how it has been labelled
        if all_data['Number of Animals'][i] >= 1 or (all_data['Photo Type'][i] in people):
            try:  # checking if the file exists
                shutil.copy(image, 'data/training/animal')  # if it exists, copy to training/animal folder
            except FileNotFoundError:
                print(f'file {image} not found')

        elif all_data['Photo Type'][i] == 'Blank':
            try:  # checking if the file exists
                shutil.copy(image, 'data/training/blank')  # if it exists, copy to training/blank folder
            except FileNotFoundError:
                print(f'file {image} not found')

        else:
            try:  # checking if the file exists
                shutil.copy(image, 'data/unlabelled')  # if it exists, copy to unlabelled folder
            except FileNotFoundError:
                print(f'file {image} not found')


process_data()
