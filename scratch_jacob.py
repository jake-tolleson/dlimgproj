import pandas as pd
import numpy as np
import tensorflow as tf

# Getting image directories
train_img_path = '/Users/jacobheinrich/Documents/GitHub/dlimgproj/SVHN/train'
test_img_path = '/Users/jacobheinrich/Documents/GitHub/dlimgproj/SVHN/test'

# Getting json file paths
train_json_path = '/Users/jacobheinrich/Documents/GitHub/dlimgproj/SVHN/train/digitStruct.json'
test_json_path = '/Users/jacobheinrich/Documents/GitHub/dlimgproj/SVHN/test/digitStruct.json'

# Crop individual numbers using openCV or something similar
def crop_digit(img, json_file, digit):
    pic = img[json_file['digitStruct'][digit]['bbox']['y1']:json_file['digitStruct'][digit]['bbox']['y2'],
                json_file['digitStruct'][digit]['bbox']['x1']:json_file['digitStruct'][digit]['bbox']['x2']]
    return(pic)

crop_digit(train_img_path, train_json_path, 0)
crop_digit(test_img_path, test_json_path, 0)

# Train CNN based on all individual number images


##maybe try paddings vs rescaling
## DATA PREPROCESSING
# Find dimensions of largest image in data


# Pad all other images to match this dimension
def pad_images(data, max_dim):
    for i in range(len(data)):
        if data['digitStruct'][i]['bbox']['height'] < max_dim:
            data['digitStruct'][i]['bbox']['height'] = max_dim
        if data['digitStruct'][i]['bbox']['width'] < max_dim:
            data['digitStruct'][i]['bbox']['width'] = max_dim
    return(data)

# Scale the data (divide by 255)
def scale_data(data):
    for i in range(len(data)):
        data['digitStruct'][i]['bbox']['x1'] = data['digitStruct'][i]['bbox']['x1'] / 255
        data['digitStruct'][i]['bbox']['x2'] = data['digitStruct'][i]['bbox']['x2'] / 255
        data['digitStruct'][i]['bbox']['y1'] = data['digitStruct'][i]['bbox']['y1'] / 255
        data['digitStruct'][i]['bbox']['y2'] = data['digitStruct'][i]['bbox']['y2'] / 255
    return(data)


# Convert labels to one-hot encodings
def convert_digit_to_array(digit):
    if digit == 0.0:
        return(np.array([1,0,0,0,0,0,0,0,0,0]))
    if digit == 1.0:
        return(np.array([0,1,0,0,0,0,0,0,0,0]))
    if digit == 2.0:
        return(np.array([0,0,1,0,0,0,0,0,0,0]))
    if digit == 3.0:
        return(np.array([0,0,0,1,0,0,0,0,0,0]))
    if digit == 4.0:
        return(np.array([0,0,0,0,1,0,0,0,0,0]))
    if digit == 5.0:
        return(np.array([0,0,0,0,0,1,0,0,0,0]))
    if digit == 6.0:
        return(np.array([0,0,0,0,0,0,1,0,0,0]))
    if digit == 7.0:
        return(np.array([0,0,0,0,0,0,0,1,0,0]))
    if digit == 8.0:
        return(np.array([0,0,0,0,0,0,0,0,1,0]))
    if digit == 9.0:
        return(np.array([0,0,0,0,0,0,0,0,0,1]))

convert_digit_to_array(test_data['digitStruct'][0]['label'])


train_data['label'] = 0
train_data = train_data.astype('object')

for index, label in enumerate(train_data['boxes']):
    digit_list = []
    for digit in label:
        digit_list.append(convert_digit_to_array(digit['label']))
    train_data['label'].loc[index] = digit_list


# Some of these labels seem incorrect...
train_data.iloc[33396,0] # label shows 19
train_data.iloc[33396,1] # image looks like a single 8 or 9

# Finding max length digit
max_length = 0

for label in train_data['label']:
    if len(label) > max_length:
        max_length = len(label) 
        
max_length # max length is 6. Means we need 7 output layers. 1 for each of 6 digits and the last will represent length


# Now lets do some image preprocessing
# We need to go through and find the largest image in the data and pad all others to that size...
max_dim = find_max_dim(train_data)
train_data = pad_images(train_data, max_dim)
test_data = pad_images(test_data, max_dim)
