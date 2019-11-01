
# Introduction
This repository contains a simple implementation of a net that learns to drive a car in a Unity simulation environment


# Training data

Training data contains total 6 laps for each track <br> 
I first recorded three laps on track one using center lane driving and some recoveries, then three  in the opposite way. The same happened for the other track


# Data Augmentation and preprocess

The overall strategy for deriving a model architecture was to start by preprocessing the data. I've started by cropping the image and converting it to YUV like the NVIDIA document best practice. 
  Bearing in mind that data is never enough, I've implemented a generator which augments randomly the images by:
* randomly flip some images to keep balance of left turn and right turn 
* randomly adjust brightness of some pics 
* add some random horizontal and vertical shift and adjust the steering angle accordingly. 

```python
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
```

# Model
The final model architecture (model.py lines 25-43) consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 66x200x3 RGB image   							| 
| Lambda         		| Normalizing and adjusting mean				|  
| Convolution 5x5x24   	| 2x2 stride, same padding                      |
| RELU					|												|
| Convolution 5x5x36   	| 2x2 stride, same padding                      |
| RELU					|												|
| Convolution 5x5x48   	| 2x2 stride, same padding                      |
| RELU					|												|
| Convolution 3x3x64   	| 2x2 stride, same padding                      |
| RELU					|												|
| Convolution 3x3x64   	| 3x3 stride, same padding                      |
| RELU					|												|
| Flatten               | Flattening to link with Dense layer           |
| Dropout               | Droput with 50% probability                   |
| Dense         	    | 100 neurons  									|
| RELU					|												|
| Dense         	    | 50 neurons  									|
| RELU					|												|
| Dense         	    | 10 neurons  									|
| Dense         	    | 1 neurons  									|

# Launch

To launch the model, simply run before the Unity environment at https://github.com/udacity/self-driving-car-sim and then from the command line `python drive.py model.h5`

