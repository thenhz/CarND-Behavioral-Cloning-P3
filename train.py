from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from math import ceil
import pandas as pd
from scipy import ndimage
from model import *

batch_size = 32

track1_df = pd.read_csv('/opt/data/train_hd/driving_log.csv', header=None)
track1_df = track1_df.rename(
    columns={0: 'Center Image', 1: 'Left Image', 2: 'Right Image', 3: 'Steering Angle', 4: 'Throttle', 5: 'Break',
             6: 'Speed'})

train_samples, valid_samples = train_test_split(track1_df, test_size=0.2)

train_generator = utils.generator(train_samples, bs = batch_size)
valid_generator = utils.generator(valid_samples, bs = batch_size)

model = get_model()

# from keras.utils.training_utils import multi_gpu_model
# model = multi_gpu_model(model, gpus=2)

model.compile(loss="mse", optimizer=Adam())

es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=5)
cp = ModelCheckpoint('model3.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = model.fit_generator(train_generator,
                              steps_per_epoch=ceil(len(train_samples) / batch_size),
                              validation_data=valid_generator,
                              validation_steps=ceil(len(valid_samples) / batch_size),
                              epochs=60, callbacks=[es, cp])
