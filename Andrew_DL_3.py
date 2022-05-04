from cv2 import IMREAD_GRAYSCALE
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import cv2

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



# Creating digit labels for train data
train_data['label'] = 0
train_data = train_data.astype('object')

for index, label in enumerate(train_data['boxes']):
    digit_list = []
    for digit in label:
        digit_list.append(convert_digit_to_array(digit['label']))
    train_data['label'].loc[index] = digit_list



# Creating digit labels for test data
test_data['label'] = 0
test_data = test_data.astype('object')

for index, label in enumerate(test_data['boxes']):
    digit_list = []
    for digit in label:
        digit_list.append(convert_digit_to_array(digit['label']))
    test_data['label'].loc[index] = digit_list


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

# We first need to crop down all images to specified BB sizes
train_data.boxes[0] # we need min(top) and min(left) from all BBs. Then we do top+height and left+width to get the bottom right x,y coords



# Finding BB for train Data
train_data['BB'] = 0
train_data = train_data.astype('object')

for index, label in enumerate(train_data.boxes):
    topLeftX = 0
    topLeftY = 0
    bottomRightX = 0
    bottomRightY = 0
    
    for digit in label:
        if digit['top'] > topLeftY:
            topLeftY = digit['top']
        if digit['left'] > topLeftX:
            topLeftX = digit['left']
        if digit['top'] + digit['height'] > bottomRightY:
            bottomRightY = digit['top'] + digit['height']
        if digit['left'] + digit['width'] > bottomRightX:
            bottomRightX = digit['left'] + digit['width']
        
        train_data['BB'].loc[index] = [topLeftX,topLeftY,bottomRightX,bottomRightY]
        

# Finding BB for Test Data
test_data['BB'] = 0
test_data = test_data.astype('object')

for index, label in enumerate(test_data.boxes):
    topLeftX = 0
    topLeftY = 0
    bottomRightX = 0
    bottomRightY = 0
    
    for digit in label:
        if digit['top'] > topLeftY:
            topLeftY = digit['top']
        if digit['left'] > topLeftX:
            topLeftX = digit['left']
        if digit['top'] + digit['height'] > bottomRightY:
            bottomRightY = digit['top'] + digit['height']
        if digit['left'] + digit['width'] > bottomRightX:
            bottomRightX = digit['left'] + digit['width']
        
        test_data['BB'].loc[index] = [topLeftX,topLeftY,bottomRightX,bottomRightY]





# We need to go through and find the largest image in the data and pad all others to that size...
train_max_height = 0
train_max_width = 0

for image in range(len(train_data)):

    width = train_data.BB[image][2] - train_data.BB[image][0]
    height = train_data.BB[image][3] - train_data.BB[image][1]
    
    if width > train_max_width:
        train_max_width = width
    if height > train_max_height:
        train_max_height = height

print(train_max_height) # 403 - train
print(train_max_width) # 207 -train

test_max_height = 0
test_max_width = 0

for image in range(len(test_data)):

    width = test_data.BB[image][2] - test_data.BB[image][0]
    height = test_data.BB[image][3] - test_data.BB[image][1]
    
    if width > test_max_width:
        test_max_width = width
    if height > test_max_height:
        test_max_height = height

print(test_max_height) # 208 - test
print(test_max_width) # 133 - test



# So we need to crop to the bounding pad all images to 403 x 207
train_cropped_path = 'C:/Users/Andrew/Desktop/Spring 2022/BZAN 554/Group Assignment/3/SVHN/train/cropped/'
test_cropped_path = 'C:/Users/Andrew/Desktop/Spring 2022/BZAN 554/Group Assignment/3/SVHN/test/cropped/'

for image in range(len(test_data)):
    img_path = os.path.join(train_img_path, train_data['filename'].loc[image])
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    cropped_image = image[train_data.BB[image][0]:train_data.BB[image][2],train_data.BB[image][1]:train_data.BB[image][3]]
    
    old_image_width = train_data.BB[image][2] - train_data.BB[image][0]
    old_image_height = train_data.BB[image][3] - train_data.BB[image][1]
    
    # create new image of desired size and color (white) for padding
    new_image_width = max([train_max_width,test_max_width])
    new_image_height = max([train_max_height,test_max_height])
    color = (255,255,255)
    result = np.full((new_image_height,new_image_width), color, dtype=np.uint8)

    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image
    result[y_center:y_center+old_image_height, x_center:x_center+old_image_width] = cropped_image

    # save result
    cv2.imwrite(os.path.join(train_cropped_path , train_data['filename'].loc[image]), result)


img_path = os.path.join(train_img_path, train_data['filename'].loc[0])
image = cv2.imread(img_path)
cropped_image = image[int(train_data.BB[0][0]):int(train_data.BB[0][2]),int(train_data.BB[0][1]):int(train_data.BB[0][3])]

old_image_width = int(train_data.BB[0][2] - train_data.BB[0][0])
old_image_height = int(train_data.BB[0][3] - train_data.BB[0][1])

# create new image of desired size and color (white) for padding
new_image_width = int(max([train_max_width,test_max_width]))
new_image_height = int(max([train_max_height,test_max_height]))
color = (255,255,255)
result = np.full((new_image_width,new_image_height,3), color, dtype=np.uint8)

# compute center offset
x_center = int((new_image_width - old_image_width) // 2)
y_center = int((new_image_height - old_image_height) // 2)

# copy img image into center of result image
result[y_center:y_center+old_image_height, x_center:x_center+old_image_width] = cropped_image

# save result
cv2.imwrite(os.path.join(train_cropped_path , train_data['filename'].loc[image]), result)