from cv2 import IMREAD_GRAYSCALE
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt

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
    if digit == 10.0:
        return(np.array([1,0,0,0,0,0,0,0,0,0]))



# Finding max length digit
max_length = 0

for label in train_data['label']:
    if len(label) > max_length:
        max_length = len(label) 
        
max_length # max length is 6. Means we need 7 output layers. 1 for each of 6 digits and the last will represent length


# Creating digit labels for train data
train_data['label'] = 0
train_data = train_data.astype('object')

for index, label in enumerate(train_data['boxes']):
    digit_list = []
    for digit in label:
        digit_list.append(convert_digit_to_array(digit['label']))
    while len(digit_list) < max_length:
        digit_list.append(np.array([0,0,0,0,0,0,0,0,0,0]))
    digit_list.append(convert_digit_to_array(len(label)))
    train_data['label'].loc[index] = digit_list

#from itertools import chain
#np.array(list(chain.from_iterable(train_data['label'][0])))

# Creating digit labels for test data
test_data['label'] = 0
test_data = test_data.astype('object')

for index, label in enumerate(test_data['boxes']):
    digit_list = []
    for digit in label:
        digit_list.append(convert_digit_to_array(digit['label']))
    while len(digit_list) < max_length:
        digit_list.append(np.array([0,0,0,0,0,0,0,0,0,0]))
    digit_list.append(convert_digit_to_array(len(label)))
    test_data['label'].loc[index] = digit_list


# Some of these labels seem incorrect...
train_data.iloc[33396,0] # label shows 19
train_data.iloc[33396,1] # image looks like a single 8 or 9


# Now lets do some image preprocessing

# We first need to crop down all images to specified BB sizes
train_data.boxes[0] # we need min(top) and min(left) from all BBs. Then we do top+height and left+width to get the bottom right x,y coords



# Finding BB for train Data
train_data['BB'] = 0
train_data = train_data.astype('object')
train_data.boxes[0]
for index, label in enumerate(train_data.boxes):
    topLeftX = np.Inf
    topLeftY = np.Inf
    bottomRightX = 0
    bottomRightY = 0
    
    for digit in label:
        if digit['top'] < topLeftY:
            topLeftY = digit['top']
        if digit['left'] < topLeftX:
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
    topLeftX = np.Inf
    topLeftY = np.Inf
    bottomRightX = 0
    bottomRightY = 0
    
    for digit in label:
        if digit['top'] < topLeftY:
            topLeftY = digit['top']
        if digit['left'] < topLeftX:
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

print(train_max_height) # 423 - train
print(train_max_width) # 616 -train

test_max_height = 0
test_max_width = 0

for image in range(len(test_data)):

    width = test_data.BB[image][2] - test_data.BB[image][0]
    height = test_data.BB[image][3] - test_data.BB[image][1]
    
    if width > test_max_width:
        test_max_width = width
    if height > test_max_height:
        test_max_height = height

print(test_max_height) # 222 - test
print(test_max_width) # 206 - test









train_cropped_path = 'C:/Users/Andrew/Desktop/Spring 2022/BZAN 554/Group Assignment/3/SVHN/train/cropped/'
test_cropped_path = 'C:/Users/Andrew/Desktop/Spring 2022/BZAN 554/Group Assignment/3/SVHN/test/cropped/'


######### ONLY NEED TO RUN THIS ONCE

# So we need to crop to the bounding pad all images to 423 x 616

for idx in range(len(train_data)):
    img_path = os.path.join(train_img_path, train_data['filename'].loc[idx])
    image = cv2.imread(img_path)
    cropped_image = image[int(train_data.BB[idx][1]):int(train_data.BB[idx][3]),int(train_data.BB[idx][0]):int(train_data.BB[idx][2])]
    old_image_height, old_image_width, channels = cropped_image.shape

    # create new image of desired size and color (white) for padding
    new_image_width = int(max([train_max_width,test_max_width]))
    new_image_height = int(max([train_max_height,test_max_height]))
    color = (255,255,255)
    result = np.full((new_image_height,new_image_width,3), color, dtype=np.uint8)

    # compute center offset
    x_center = int((new_image_width - old_image_width) // 2)
    y_center = int((new_image_height - old_image_height) // 2)

    # copy img image into center of result image
    result[y_center:y_center+old_image_height, x_center:x_center+old_image_width] = cropped_image

    # save result
    cv2.imwrite(os.path.join(train_cropped_path , train_data['filename'].loc[idx]), result)


for idx in range(len(test_data)):
    img_path = os.path.join(test_img_path, test_data['filename'].loc[idx])
    image = cv2.imread(img_path)
    cropped_image = image[int(test_data.BB[idx][1]):int(test_data.BB[idx][3]),int(test_data.BB[idx][0]):int(test_data.BB[idx][2])]
    old_image_height, old_image_width, channels = cropped_image.shape

    # create new image of desired size and color (white) for padding
    new_image_width = int(max([train_max_width,test_max_width]))
    new_image_height = int(max([train_max_height,test_max_height]))
    color = (255,255,255)
    result = np.full((new_image_height,new_image_width,3), color, dtype=np.uint8)

    # compute center offset
    x_center = int((new_image_width - old_image_width) // 2)
    y_center = int((new_image_height - old_image_height) // 2)

    # copy img image into center of result image
    result[y_center:y_center+old_image_height, x_center:x_center+old_image_width] = cropped_image

    # save result
    cv2.imwrite(os.path.join(test_cropped_path , test_data['filename'].loc[idx]), result)


# Now we have all the data in the format that we want it to be in.



###### END OF ONLY NEED TO RUN THIS ONCE














train_cropped_path = 'C:/Users/Andrew/Desktop/Spring 2022/BZAN 554/Group Assignment/3/SVHN/train/cropped/'
test_cropped_path = 'C:/Users/Andrew/Desktop/Spring 2022/BZAN 554/Group Assignment/3/SVHN/test/cropped/'


# Need to make train and test values
train = train_data.drop(columns=['boxes','BB'])

# creating a column to hold images
train['image'] = 0
train = train.astype('object')

for idx,img in enumerate(train['filename']):
    img_path = os.path.join(train_cropped_path, img)
    train['image'].loc[idx] = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.0
    #train['label'].loc[idx] = tf.convert_to_tensor(train['label'][idx])

test = test_data.drop(columns=['boxes','BB'])
test['image'] = 0
test = test.astype('object')

for idx,img in enumerate(test['filename']):
    img_path = os.path.join(test_cropped_path, img)
    test['image'].loc[idx] = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.0
    test['label'].loc[idx] = tf.convert_to_tensor(test['label'][idx])


inputs = tf.keras.layers.Input(shape=(int(max([train_max_height,test_max_height])),int(max([train_max_width,test_max_width])),1), name='input') 
x = tf.keras.layers.Conv2D(filters=64,kernel_size = 7, strides = 1, padding = "same", activation = "relu")(inputs)
x = tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2, padding = "valid")(x)
x = tf.keras.layers.Conv2D(filters=128,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(x)
x = tf.keras.layers.Conv2D(filters=128,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(x)
x = tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2, padding = "valid")(x)
x = tf.keras.layers.Conv2D(filters=256,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(x)
x = tf.keras.layers.Conv2D(filters=256,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(x)
x = tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2, padding = "valid")(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation = 'relu')(x)
x = tf.keras.layers.Dense(64, activation = 'relu')(x)
yhat_1 = tf.keras.layers.Dense(10, activation = 'softmax')(x)
yhat_2 = tf.keras.layers.Dense(10, activation = 'softmax')(x)
yhat_3 = tf.keras.layers.Dense(10, activation = 'softmax')(x)
yhat_4 = tf.keras.layers.Dense(10, activation = 'softmax')(x)
yhat_5 = tf.keras.layers.Dense(10, activation = 'softmax')(x)
yhat_6 = tf.keras.layers.Dense(10, activation = 'softmax')(x)
yhat_length = tf.keras.layers.Dense(10, activation = 'softmax')(x)

model = tf.keras.Model(inputs = inputs, outputs = [yhat_1, yhat_2, yhat_3, yhat_4, yhat_5, yhat_6, yhat_length])
model.summary()
#Compile model
model.compile(loss = 'categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001))

#Fit model
model.fit(x=train['image'][0],y=train['label'][0], batch_size=1, epochs=1) 
train['image'][3].shape
train['label'][0]

#Compute multiclass accuray
yhat = model.predict(x=X_test)
yhat_sparse = [int(np.where(yhat_sub ==np.max(yhat_sub))[0]) for yhat_sub in yhat]
y_test
sum(yhat_sparse == y_test) / len(y_test)


