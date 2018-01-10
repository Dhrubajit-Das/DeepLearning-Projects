import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
root_dir = os.path.abspath('../..')
data_dir = os.path.join(root_dir, 'data')
sub_dir = os.path.join(root_dir, 'sub')

# loading label files - csv
train = pd.read_csv(os.path.join(data_dir, 'Train',".......training labels- csv file url........"))

#to plot an image
seed = 128
rng = np.random.RandomState(seed)

import pylab
from scipy.misc import imread
img_name = rng.choice(train.filename)
filepath = os.path.join(data_dir, 'Train', 'Images', '....training images folder url......', img_name)

img = imread(filepath)

pylab.imshow(img, cmap='gray')
pylab.axis('off')
pylab.show()

#Storing all our images as numpy arrays.....
#Train data...
from scipy.misc import imread
temp = []
for img_name in train.filename:
    image_path = os.path.join(data_dir, 'Train', 'Images', '....training images folder url......', img_name)
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    temp.append(img)
    
train_x = np.stack(temp)
train_x /= 255.0
train_x = train_x.reshape(-1, 784).astype('float32')


#Training Labels..
import keras
train_y = keras.utils.np_utils.to_categorical(train.label.values)

#splitting the data
split_size = int(train_x.shape[0]*0.7)
train_x, val_x = train_x[:split_size], train_x[split_size:]
train_y, val_y = train_y[:split_size], train_y[split_size:]


# reshape data to fit into the CNN
train_x_temp = train_x.reshape(-1, 28, 28, 1)
val_x_temp = val_x.reshape(-1, 28, 28, 1)

# define vars
from keras.layers import InputLayer,MaxPooling2D,Flatten,Convolution2D, Dense
from keras.models import Sequential


#Model 1 : baseline model
model1 = Sequential([
 InputLayer(input_shape=(28, 28, 1)),

 Convolution2D(32, 3, 3, activation='relu'),
 MaxPooling2D(pool_size=(2, 2)),
 Flatten(),
 Dense(output_dim=50, activation='relu'),
 Dense(output_dim=10, activation='softmax'),
])
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

cnn1 = model1.fit(train_x_temp, train_y, nb_epoch=15, batch_size=128, validation_data=(val_x_temp, val_y))



#Model 2 : smaller model
model2 = Sequential([
 InputLayer(input_shape=(28, 28, 1)),

 Convolution2D(18, 3, 3, activation='relu'),
 MaxPooling2D(pool_size=(2, 2)),
 Flatten(),
 Dense(output_dim=50, activation='relu'),
 Dense(output_dim=10, activation='softmax'),
])
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

cnn2 = model2.fit(train_x_temp, train_y, nb_epoch=15, batch_size=128, validation_data=(val_x_temp, val_y))



#Model 3 : wider model
model3 = Sequential([
 InputLayer(input_shape=(28, 28, 1)),

 Convolution2D(64, 3, 3, activation='relu'),
 MaxPooling2D(pool_size=(2, 2)),
 Flatten(),
 Dense(output_dim=50, activation='relu'),
 Dense(output_dim=10, activation='softmax'),
])
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

cnn3 = model3.fit(train_x_temp, train_y, nb_epoch=15, batch_size=128, validation_data=(val_x_temp, val_y))



#Model 4 : deep model
model4 = Sequential([
 InputLayer(input_shape=(28, 28, 1)),

 Convolution2D(64, 3, 3, activation='relu'),
 MaxPooling2D(pool_size=(2, 2)),
 Convolution2D(32, 3, 3, activation='relu'),
 MaxPooling2D(pool_size=(2, 2)),
 Flatten(),
 Dense(output_dim=50, activation='relu'),
 Dense(output_dim=10, activation='softmax'),
])
model4.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

cnn4 = model4.fit(train_x_temp, train_y, nb_epoch=15, batch_size=128, validation_data=(val_x_temp, val_y))



#Model 5: CNN increasing epochs size
model5 = Sequential([
 InputLayer(input_shape=(28, 28, 1)),

 Convolution2D(64, 3, 3, activation='relu'),
 MaxPooling2D(pool_size=(2, 2)),
 Convolution2D(32, 3, 3, activation='relu'),
 MaxPooling2D(pool_size=(2, 2)),
 Flatten(),
 Dense(output_dim=50, activation='relu'),
 Dense(output_dim=10, activation='softmax'),
])
model5.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

cnn5 = model5.fit(train_x_temp, train_y, nb_epoch=50, batch_size=128, validation_data=(val_x_temp, val_y))




#Model 6: changing activation function for output layer
model6 = Sequential([
 InputLayer(input_shape=(28, 28, 1)),

 Convolution2D(64, 3, 3, activation='relu'),
 MaxPooling2D(pool_size=(2, 2)),
 Convolution2D(32, 3, 3, activation='relu'),
 MaxPooling2D(pool_size=(2, 2)),
 Flatten(),
 Dense(output_dim=50, activation='relu'),
 Dense(output_dim=10, activation='sigmoid'),
])
model6.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

cnn6 = model6.fit(train_x_temp, train_y, nb_epoch=50, batch_size=128, validation_data=(val_x_temp, val_y))



#Model 7: making a much deeper model
model7 = Sequential([
 InputLayer(input_shape=(28, 28, 1)),

 Convolution2D(64, 3, 3, activation='relu'),
 MaxPooling2D(pool_size=(2, 2)),
 Convolution2D(32, 3, 3, activation='relu'),
 MaxPooling2D(pool_size=(2, 2)),
 Convolution2D(32, 3, 3, activation='relu'),
 MaxPooling2D(pool_size=(2, 2)),
 Flatten(),
 Dense(output_dim=50, activation='relu'),
 Dense(output_dim=50, activation='relu'),
 Dense(output_dim=10, activation='sigmoid'),
])
model7.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

cnn7 = model7.fit(train_x_temp, train_y, nb_epoch=50, batch_size=128, validation_data=(val_x_temp, val_y))


#Model 8: using sgd optimizer
model8 = Sequential([
 InputLayer(input_shape=(28, 28, 1)),

 Convolution2D(64, 3, 3, activation='relu'),
 MaxPooling2D(pool_size=(2, 2)),
 Convolution2D(32, 3, 3, activation='relu'),
 MaxPooling2D(pool_size=(2, 2)),
 Convolution2D(32, 3, 3, activation='relu'),
 MaxPooling2D(pool_size=(2, 2)),
 Flatten(),
 Dense(output_dim=50, activation='relu'),
 Dense(output_dim=50, activation='relu'),
 Dense(output_dim=10, activation='sigmoid'),
])
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.5, nesterov=True)
model8.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

cnn8 = model8.fit(train_x_temp, train_y, nb_epoch=50, batch_size=128, validation_data=(val_x_temp, val_y))


# plotting loss and accuracy curves
# Loss curve
plt.figure(figsize=[8,6])
plt.plot(cnn3.history['loss'],'r',linewidth=3.0)
plt.plot(cnn3.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=8)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curve',fontsize=16)
 
# Accuracy Curve
plt.figure(figsize=[8,6])
plt.plot(cnn3.history['acc'],'r',linewidth=3.0)
plt.plot(cnn3.history['val_acc'],'b',linewidth=3.0)
plt.axhline(y=.99, xmin=0.0, xmax=14, linewidth = 2, color='green')
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=10, loc=4)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curve',fontsize=16)


# predicting on test set and converting the results into a csv file
test = pd.read_csv(os.path.join(data_dir, 'Train',".......test labels- csv file url........"))

#Storing all our test images as numpy arrays.....
temp1 = []
for img_name in test.filename:
    image_path = os.path.join(data_dir, 'Train', 'Images', '....test images folder url......', img_name)
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    temp1.append(img)
    
test_x = np.stack(temp1)
test_x /= 255.0
test_x = test_x.reshape(-1, 784).astype('float32')

sample_submission.to_csv(os.path.join(sub_dir, 'test file url'), index=False)
pred_test = test_x.reshape(-1, 28, 28, 1)
prediction = model.predict_classes(pred_test)
sample_submission.filename = test.filename; sample_submission.label = prediction
