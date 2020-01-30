import os
import shutil
import pandas as pd
import re
from math import log10
import random


def process_data(train_pct=70, valid_pct=15, test_pct=15):
    """
    This function will create the directories data/training, data/training/animal,
    data/training/blank, and data/unlabelled and will populate these directories with
    images. It will also create directories for validation and testing data. First, the function
    will put all images in the data/training folders, and then it will randomly move some images to
    the validation and testing data sets. The number of images that will be moved is determined by
    the percentages passed into this function.

    :param train_pct: The percentage of images to be used for training.
    :param valid_pct: The percentage of images to be used for validation.
    :param test_pct: The percentage of images to be used for testing.
    """

    if train_pct + valid_pct + test_pct != 100:
        print('Invalid Train/Valid/Test Split values')
        exit()

    random_seed = 398

    # Here I create the necessary training folders
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

    # Here I create the necessary validation folders
    try:
        os.mkdir('data/validation')
    except OSError:
        print('Creation of training directory failed')
    else:
        print('Creation of training directory successful')

    try:
        os.mkdir('data/validation/animal')
    except OSError:
        print("Creation of animal directory failed")
    else:
        print('Creation of animal Directory Successful')

    try:
        os.mkdir('data/validation/blank')
    except OSError:
        print("Creation of blank directory failed")
    else:
        print('Creation of blank Directory Successful')

    # Here I create the necessary test folders
    try:
        os.mkdir('data/test')
    except OSError:
        print('Creation of test directory failed')
    else:
        print('Creation of test directory successful')

    try:
        os.mkdir('data/test/animal')
    except OSError:
        print("Creation of animal directory failed")
    else:
        print('Creation of animal Directory Successful')

    try:
        os.mkdir('data/test/blank')
    except OSError:
        print("Creation of blank directory failed")
    else:
        print('Creation of blank Directory Successful')

    # Here I create a folder for all of the unlabelled images
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
        if '.xlsx' in file or '.csv' in file:  # If .xlsx is in file name, the file contains data
            label_files.append(file)
        else:  # If not, then it will be a directory with images
            image_dirs.append(file)

    # Create a dataframe of all labelled data
    all_data = pd.DataFrame()
    for file in label_files:  # loop through all .xlsx files
        if '.xlsx' in file:
            df = pd.read_excel(os.path.join('data/MuleDeetData/', file))  # read in next file
            all_data = all_data.append(df, ignore_index=True)  # append to dataframe
        elif '.csv' in file:
            df = pd.read_csv(os.path.join('data/MuleDeetData/', file))
            all_data = all_data.append(df, ignore_index=True)

    people = ['Start', 'Setup/Pickup', 'End']  # List of labels that identify people
    for i in range(len(all_data)):  # Loop through all images in datafile

        # Get the name of the folder that contains the image
        folder = re.sub('_(\d)+\.(?i)JPG$', '', all_data['Raw Name'][i])

        # Get the name and location of the image
        image = os.path.join('data/MuleDeetData', folder, all_data['Raw Name'][i])

        # Check if the image has been labelled and how it has been labelled
        if all_data['Number of Animals'][i] >= 1 or \
                (all_data['Photo Type'][i] in people) or \
                (all_data['Photo Type'][i] == 'animal'):
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

    # This is to store a list of all the available images
    animals = os.listdir('data/training/animal')
    blanks = os.listdir('data/training/blank')

    # Checking the number of Animal and Blank Images
    num_animals = len(animals)
    num_blanks = len(blanks)

    # Summarize the number of images available
    print('=' * (44 + int(log10(num_animals + num_blanks))))
    print(f'Total Number of images available: {num_blanks + num_animals}')
    print(f'Total Number of images containing animals: {num_animals}')
    print(f'Total Number of blank images: {num_blanks}')
    print('=' * (44 + int(log10(num_animals + num_blanks))))

    # Get the number of images with animals for training, validation, and testing sets
    num_train_animals = int(round(train_pct * num_animals / 100))
    num_valid_animals = int(round(valid_pct * num_animals / 100))
    num_test_animals = int(round(test_pct * num_animals / 100))

    # Make sure that train, val, and test sum to total number of images. If not, gve extra to train
    if num_train_animals + num_valid_animals + num_test_animals != num_animals:
        num_train_animals = num_animals - (num_valid_animals + num_test_animals)

    # Summarize the number of images with animals in them according to each data set
    print('=' * (44 + int(log10(num_animals + num_blanks))))
    print(f'Number of Training Animals: {num_train_animals}')
    print(f'Number of Validation Animals: {num_valid_animals}')
    print(f'Number of Test Animals: {num_test_animals}')
    print('=' * (44 + int(log10(num_animals + num_blanks))))

    #  Get the number of Blanks for each of the data sets
    num_train_blanks = int(round(train_pct * num_blanks / 100))
    num_valid_blanks = int(round(valid_pct * num_blanks / 100))
    num_test_blanks = int(round(test_pct * num_blanks / 100))

    # Make sure the number of images of all sets sums to total; if not, give extras to training.
    if num_train_blanks + num_valid_blanks + num_test_blanks != num_blanks:
        num_train_blanks = num_blanks - (num_valid_blanks + num_test_blanks)

    # Summarize the number of blank images in each of the data sets.
    print('=' * (44 + int(log10(num_blanks + num_blanks))))
    print(f'Number of Training blanks: {num_train_blanks}')
    print(f'Number of Validation blanks: {num_valid_blanks}')
    print(f'Number of Test blanks: {num_test_blanks}')
    print('=' * (44 + int(log10(num_blanks + num_blanks))))

    # Store an array of possible animal indices and blank image indices
    animal_indices = list(range(num_animals))
    blanks_indices = list(range(num_blanks))

    # Shuffle indices arrays so that we don't get train images from one location and test from another
    random.seed(random_seed)
    random.shuffle(animal_indices)
    random.shuffle(blanks_indices)

    # Set aside the first <Number of validation animal images> indices for validation set
    valid_animals_indices = animal_indices[:num_valid_animals]

    # Set aside the next <Number of testing animal images> indices for testing set
    test_animals_indices = animal_indices[num_valid_animals:(num_valid_animals + num_test_animals)]

    # Do the same for the blank images
    valid_blanks_indices = blanks_indices[:num_valid_blanks]
    test_blanks_indices = blanks_indices[num_valid_blanks:(num_valid_blanks + num_test_blanks)]

    # Loop through the indices for validation and set aside those images into an new directory
    print('Moving validation images......', end='')
    for i in valid_animals_indices:
        try:
            shutil.move(os.path.join('data/training/animal', animals[i]),
                        os.path.join('data/validation/animal', animals[i]))
        except FileNotFoundError:
            print(f"File {os.path.join('data/training/animal', animals[i])} not found")

    for i in valid_blanks_indices:
        try:
            shutil.move(os.path.join('data/training/blank', blanks[i]),
                        os.path.join('data/validation/blank', blanks[i]))
        except FileNotFoundError:
            print(f"File {os.path.join('data/training/blank', blanks[i])} not found")

    print('Completed')

    print('Moving testing images......', end='')
    for i in test_animals_indices:
        try:
            shutil.move(os.path.join('data/training/animal', animals[i]),
                        os.path.join('data/test/animal', animals[i]))
        except FileNotFoundError:
            print(f"File {os.path.join('data/training/animal', animals[i])} not found.")

    for i in test_blanks_indices:
        try:
            shutil.move(os.path.join('data/training/blank', blanks[i]),
                        os.path.join('data/test/blank', blanks[i]))
        except FileNotFoundError:
            print(f"File {os.path.join('data/training/blank', blanks[i])} not found. ")

    print('Completed')

    # Final test to see if everything worked. Leave uncommented.
    # train_animal_size = len(os.listdir('data/training/animal'))
    # train_blanks_size = len(os.listdir('data/training/blank'))
    #
    # valid_animal_size = len(os.listdir('data/validation/animal'))
    # valid_blanks_size = len(os.listdir('data/validation/blank'))
    #
    # test_animal_size = len(os.listdir('data/test/animal'))
    # test_blanks_size = len(os.listdir('data/test/blank'))
    #
    # print(f'Final Number of Training Observations: {train_animal_size} animals and {train_blanks_size} '
    #       f'blanks')
    # print(f'Final Number of Validation Observations: {valid_animal_size} animals and {valid_blanks_size} '
    #       f'blanks')
    # print(f'Final Number of Testing Observations: {test_animal_size} animals and {test_blanks_size} '
    #       f'blanks ({test_animal_size / num_animals} and {test_blanks_size / num_blanks})')


process_data()
