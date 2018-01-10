import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

seed = 128
rng = np.random.RandomState(seed)

# setting up the folder directories.
import os
root_dir = os.path.abspath("../..")
data_dir = os.path.join(root_dir, 'data')
sub_dir = os.path.join(root_dir, 'sub')


# extract filename from images
import cv2
def extract_labels_from_image(folder):
    images = []
    name = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            #images.append(img)  # to load the images
            name.append(filename) # to load the image names
    return name

train_images_labels = extract_labels_from_image("C:\\Users\\DHRUBAJIT\\Desktop\\Kaggle Competition\\Image_classification\\8000\\train")

# put the labels in a dataframe.
train_df1 = pd.DataFrame(train_images_labels)
train_df1.rename(columns={0 : 'name'}, inplace=True)
train_df1['label'] = train_df1['name'].apply(lambda x: x[0:3])


# load images, resize and map them to their respective labels.
from scipy.misc import imread, imresize
tempp1 = []
for img_name in train_df1.name:
    img_path = os.path.join(data_dir,'train',"C:\\Users\\DHRUBAJIT\\Desktop\\Kaggle Competition\\Image_classification\\8000\\train",img_name)
    img = imread(img_path, flatten = False) #set flatten = True: to convert RGB images to gray-scale
    img = imresize(img, (64,64,3)) 
    img = img.astype('float32')
    tempp1.append(img)

train_x = np.stack(tempp1)
train_x /= 255.0
train_x = train_x.reshape(-1,64,64,3).astype('float32') 

# Preprocessing Target Labels..
import keras
train_y = train_df1.label.values
train_yy = pd.DataFrame(train_y)
train_yy.rename(columns = {0: 'labels'}, inplace=True)
val_map = {'cat': 0, 'dog': 1}
train_yy['labels'] = train_yy['labels'].map(val_map)
test_y = train_yy['labels']


#split data into training and test set
split_size1 = int(train_x.shape[0]*0.8)
train_x1, val_x1 = train_x[:split_size1], train_x[split_size1:]
train_y1, val_y1 = test_y[:split_size1], test_y[split_size1:]


# CNN
train_x_temp = train_x1.reshape(-1, 64,64,3)
val_x_temp = val_x1.reshape(-1, 64,64, 3)

from keras.layers import InputLayer,MaxPooling2D,Flatten,Convolution2D, Dense
from keras.models import Sequential

model = Sequential()
model.add(Convolution2D(32, 3,3, input_shape=(64,64,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(output_dim = 128, activation = 'relu'))
model.add(Dense(output_dim = 1, activation = 'softmax'))
model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

model_cnn = model.fit(train_x_temp, val_x1, nb_epoch=20, batch_size=32, validation_data=(val_x_temp, val_y1))

# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(model_cnn.history['loss'],'r',linewidth=3.0)
plt.plot(model_cnn.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(model_cnn.history['acc'],'r',linewidth=3.0)
plt.plot(model_cnn.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)


