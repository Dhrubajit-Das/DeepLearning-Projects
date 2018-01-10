import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten


#Fitting CNN to the dataset
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
                                   
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory("C:\\Users\\DHRUBAJIT\\Desktop\\Udemy Courses\\Data Science Course\\Deep Learning\\2 Convolution Neural Network\\Dataset\\Convolutional_Neural_Networks\\dataset\\training_set",
                                                 target_size = (64,64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory("C:\\Users\\DHRUBAJIT\\Desktop\\Udemy Courses\\Data Science Course\\Deep Learning\\2 Convolution Neural Network\\Dataset\\Convolutional_Neural_Networks\\dataset\\test_set",
                                            target_size = (64,64),
                                            batch_size = 32,
                                            class_mode = 'binary')


# Model 1
#initializing the CNN
classifier = Sequential()
classifier.add(Convolution2D(10,3,3, input_shape=(64,64,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam',loss='binary_crossentropy',metrics = ['accuracy'])

model = classifier.fit_generator(training_set,
                         samples_per_epoch = 2000,
                         nb_epoch = 20,
                         validation_data = test_set,
                         nb_val_samples = 800)


# Model 2
#initializing the CNN
classifier2 = Sequential()
classifier2.add(Convolution2D(32,3,3, input_shape=(64,64,3), activation='relu'))
classifier2.add(MaxPooling2D(pool_size=(2,2)))

classifier2.add(Convolution2D(32,3,3, activation='relu'))
classifier2.add(MaxPooling2D(pool_size=(2,2)))

classifier2.add(Flatten())
classifier2.add(Dense(output_dim = 128, activation = 'relu'))
classifier2.add(Dense(output_dim = 1, activation = 'sigmoid'))
classifier2.compile(optimizer = 'adam',loss='binary_crossentropy',metrics = ['accuracy'])

model2 = classifier2.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 20,
                         validation_data = test_set,
                         nb_val_samples = 2000)

# plotting loss and accuracy curves
plt.figure(figsize=[8,6])
plt.plot(model2.history['loss'],'r',linewidth=3.0)
plt.plot(model2.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=8)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(model2.history['acc'],'r',linewidth=3.0)
plt.plot(model2.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=8)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
                         
                         
                