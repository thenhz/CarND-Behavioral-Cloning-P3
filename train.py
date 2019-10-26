from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from math import ceil
import augment
import sklearn
import numpy as np
from scipy import ndimage
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, Flatten, Dense, Activation, Dropout
import pandas as pd
import utils
import cv2

batch_size = 64

track1_df = pd.read_csv('./train_hd/driving_log.csv', header=None)
track1_df = track1_df.rename(
    columns={0: 'Center Image', 1: 'Left Image', 2: 'Right Image', 3: 'Steering Angle', 4: 'Throttle', 5: 'Break',
             6: 'Speed'})


def augment_img(img, steering):
    img, steering = augment.flip(img, steering)
    img, steering = augment.random_shift(img, steering)
    img = augment.random_shadow(img)
    img = augment.adjust_brightness(img)
    return img, steering


# After Augmenting image we'll crop and resize image. We'll use the image size used in https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
def get_train_test_labels(df, left_correction=0.0, right_correction=0.0):
    def get_relative_path(img_path):
        relative_path = './' + img_path.split('/')[-3] + '/' + img_path.split('/')[-2] + '/' + img_path.split('/')[-1]
        return relative_path

    images = []
    steering_angle_list = []
    for index, row in df.iterrows():
        left_img = get_relative_path(row['Left Image'])
        right_img = get_relative_path(row['Right Image'])
        center_img = get_relative_path(row['Center Image'])
        angle = float(row['Steering Angle'])

        # Adjust the left and right steering angle
        langle = angle + left_correction
        rangle = angle - right_correction

        # Read Image
        limg = cv2.cvtColor(cv2.imread(left_img), cv2.COLOR_BGR2RGB)#ndimage.imread(left_img)
        cimg = cv2.cvtColor(cv2.imread(center_img), cv2.COLOR_BGR2RGB)#ndimage.imread(center_img)
        rimg = cv2.cvtColor(cv2.imread(right_img), cv2.COLOR_BGR2RGB)#ndimage.imread(right_img)

        # Augment the Image
        limg, langle = augment_img(limg, langle)
        cimg, angle = augment_img(cimg, angle)
        rimg, rangle = augment_img(rimg, rangle)

        # Preprocess the augmented images to feed to model
        limg = utils.preprocess(limg)
        cimg = utils.preprocess(cimg)
        rimg = utils.preprocess(rimg)

        images.append(limg)
        steering_angle_list.append(langle)
        images.append(cimg)
        steering_angle_list.append(angle)
        images.append(rimg)
        steering_angle_list.append(rangle)

    return images, steering_angle_list


def generator(df, bs=32):
    total = len(df)
    while 1:
        sklearn.utils.shuffle(df)
        for offset in range(0, total, bs):
            batch = df[offset:offset + bs]
            images, angles = get_train_test_labels(batch, 0.2, 0.2)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


train_samples, valid_samples = train_test_split(track1_df, test_size=0.2)

train_generator = generator(train_samples)
valid_generator = generator(valid_samples)

model = Sequential()

model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(utils.IMG_HT, utils.IMG_WIDTH, utils.IMG_CH)))

model.add(Conv2D(24, (5, 5), activation="elu", strides=(2, 2)))
model.add(Conv2D(36, (5, 5), activation="elu", strides=(2, 2)))
model.add(Conv2D(48, (5, 5), activation="elu", strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="elu", strides=(1, 1)))
model.add(Conv2D(64, (3, 3), activation="elu", strides=(1, 1)))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Dense(1))

# from keras.utils.training_utils import multi_gpu_model
# model = multi_gpu_model(model, gpus=2)

model.compile(loss="mse", optimizer=Adam(1e-4, decay=0.0))

es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=5)
cp = ModelCheckpoint('model_best_fit.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = model.fit_generator(train_generator,
                              steps_per_epoch=ceil(len(train_samples) / batch_size),
                              validation_data=valid_generator,
                              validation_steps=ceil(len(valid_samples) / batch_size),
                              epochs=60, callbacks=[es, cp])
