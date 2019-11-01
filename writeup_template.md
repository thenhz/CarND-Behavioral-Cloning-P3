# **Behavioral Cloning** 

## Writeup Template

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
please take care that I've added a line of code to 'drive.py' since I had to introduce the preprocess step to feed my network properly

#### 3. Submission code is usable and readable

The train.py file contains the code for building the model and everything for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a 5 convolution layers with 5x5 and 3x3 filter sizes to capture different facets of the provided image. Depths are set to increase on each layer (model.py lines 29-33) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 27). There's another preprocess that I perform before feeding the network but that has been implemented in a standalone function. 

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer since adding many didn't increase the network accurancy. 

The model was trained and validated on different data sets recorded between the two tracks to ensure that the model was not overfitting . The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 48).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving inversely. Data has been gathered even on the second track to be able to feed different landscape leading to a more stable training

For details about how I preprocessed the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start by preprocessing the data. I've started by cropping the image and converting it to YUV like the NVIDIA document best practice. 
  Bearing in mind that data is never enough, I've implemented a generator which augments randomly the images by:
* randomly flip some images to keep balance of left turn and right turn 
* randomly adjust brightness of some pics 
* add some random horizontal and vertical shift and adjust the steering angle accordingly. 

Data is collected from the images coming from left, center and right camera.
Since we have only steering angle for center image, I've adjusted the steering angle for left_image by adding a tiny value  and similarely for right_image (by subtracting the same tiny value)

Talking about the model, I started with a Lambda layer to normalize the coming images. Five convolutional layers then have been applied with different kernel and strides, to capture different features from the same image. 
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. My first models where probably too simple so I've increased the number of convolutional layer to let the model be able to correcty understand the images. After some time, playing with the hyperparameters, I landed to a model that was overfitting since the loss was very low on training but did't decrease as well on validation. I've tried to add regularizers and dropouts but in the end the success came from lowering the number of concolutional used and using a dropout. 

The final step was to run the simulator to see how well the car was driving around track one. There were just one point where the car fell of the track (when I started to see the lake..probably interpreted as a road since I noticed clearly that was steering toward the lake) but was enough to train for several epochs more to get rid of this behaviour

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. I've tested even on track 2 and the car is driving very well except for one point where the car sees the road it's walking and a piece of road in the background that fool the model.

#### 2. Final Model Architecture

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


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
