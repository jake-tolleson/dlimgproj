import pandas as pd
import numpy as np
import tensorflow as tf

# Getting image directories
train_img_path = 'C:/Users/Andrew/Desktop/Spring 2022/BZAN 554/Group Assignment/3/SVHN/train/'
test_img_path = 'C:/Users/Andrew/Desktop/Spring 2022/BZAN 554/Group Assignment/3/SVHN/test/'

# Getting json file paths
train_json_path = 'C:/Users/Andrew/Desktop/Spring 2022/BZAN 554/Group Assignment/3/SVHN/train/digitStruct.json'
test_json_path = 'C:/Users/Andrew/Desktop/Spring 2022/BZAN 554/Group Assignment/3/SVHN/test/digitStruct.json'

# Crop individual numbers using openCV or something similar
# Train CNN based on all individual number images

train_data = pd.read_json(train_json_path)
test_data = pd.read_json(test_json_path)


## DATA PREPROCESSING
# Find dimensions of largest image in data
# Pad all other images to match this dimension
# Scale the data (divide by 255)

# Translate Data Labels into one-hot encodings
# 


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
