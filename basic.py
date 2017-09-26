import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
lines = []

with open('./Data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

lines = lines[0:10000]
images = []
measurements = []

for line in lines:

    path = './Data/IMG/'
    middle_path = path + line[0].split('/')[-1].split("\\")[-1]
    steer_center = float(line[3])


    image_m = cv2.imread(middle_path)
    images.append(image_m)
    image_m = cv2.flip(image_m, 1)
    images.append(image_m)
    measurements.append(steer_center)
    measurements.append(steer_center * -1.0)


x_train = np.array(images)
y_train = np.array(measurements)




from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D, Dropout
from keras.optimizers import Adam

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
#model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(6,5,5, border_mode='valid',activation='relu'))
model.add(Dropout(.5))
model.add(Conv2D(12,5,5, border_mode='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())

model.add(Dense(100))
model.add(Dropout(.25))
model.add(Dense(50))
model.add(Dense(1))
model.compile(loss='mse', optimizer=Adam(lr=.001))
model.fit(x_train, y_train, nb_epoch=2, batch_size=64, validation_split=.2)

model.save('basic.h5')