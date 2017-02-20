#**Behavioral Cloning** 

This writeup outlines my approach for Project 3 of Term 1 of the Udacity Self-Driving Car Nanodegree Program.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/image1.png "Model Mean Squared Error Loss"
[image2]: ./examples/image2.jpg "Center Lane Driving"
[image3]: ./examples/image3.jpg "Recovery Driving"
[image4]: ./examples/image4.jpg "Recovery Driving"
[image5]: ./examples/image5.jpg "Recovery Driving"
[image6]: ./examples/image6.jpg "Left Camera Image"
[image7]: ./examples/image7.jpg "Center Camera Image"
[image8]: ./examples/image8.jpg "Right Camera Image"
[image9]: ./examples/image9.jpg "Flipped Image"
[image10]: ./examples/image10.jpg "Flipped Image"
[video1]: ./examples/video1.mp4 "Autonomous Driving Video"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Required Files & Quality of Code

####1. Submission includes all required files

My project includes the following files:
* model.py containing the script to create, train and save the model
* model.h5 containing a trained convolution neural network
* drive.py (provided by Udacity) for driving the vehicle in the simulator's autonomous mode
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and drive.py file and my model.h5 file, the vehicle can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for building, training and saving the convolution neural network. The code uses a Python generator ((model.py lines 21-44), to generate data for training and validation in batches rather than storing the data in memory. The file shows the pipeline I used for training and validating the model, it is clearly organized by using functions and contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

To come up with a model architecture, I took inspiration from [this paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) from NVIDIA where they trained a convolutional neural network for a similar type of problem.

My model consists of convolution layers with 3x3 and 5×5 filter (kernel) sizes and depths between 24 and 64 (model.py lines 100-102 and 107-108).

The model includes ReLU layers to introduce nonlinearity (model.py lines 100-102, 107-108, 116, 118 and 120).

The data is normalized within the model using a Keras Lambda layer (model.py line 94).

I also used a Keras Cropping2D layer (model.py line 97) for image cropping within the model as it passes through the layer. This might be useful for choosing an area of interest that excludes the sky and the hood of the vehicle.

Here I visualize (by model.py line 127) my final model architecture:
```sh
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 65, 320, 3)    0           lambda_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 158, 24)   1824        cropping2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 77, 36)    21636       convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 37, 48)     43248       convolution2d_2[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 5, 37, 48)     0           convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 35, 64)     27712       dropout_1[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 33, 64)     36928       convolution2d_4[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 1, 33, 64)     0           convolution2d_5[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2112)          0           dropout_2[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           211300      flatten_1[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 100)           0           dense_1[0][0]
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 100)           0           dropout_3[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        activation_1[0][0]
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 50)            0           dense_2[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         activation_2[0][0]
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 10)            0           dense_3[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          activation_3[0][0]
====================================================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0
____________________________________________________________________________________________________
```

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 104, 110 and 115).

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 88 and 130-133).  The validation set helped determine if the model was over or under fitting.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 125).

I fiddled a little with the nb_epoch parameter and plotted it along with the mean squared error loss of both the training and the validation set (model.py lines 141-147).
I figured out that a nb_epoch of 7 is a good option, because then both losses are low:

![alt text][image1]

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and data augmentation.

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left and right sides of the road back to center so that the model would learn what to do if the vehicle gets off to the side of the road. These images show what a recovery looks like starting from the left side of the road steering back to the middle:

![alt text][image3]
![alt text][image4]
![alt text][image5]

The simulator captures images from three cameras mounted on the vehicle: a left, center and right camera. That’s because of the issue of recovering from being off-center.
I augmented the data set with the images from these three cameras.
However, the simulator only records the steering angle of an image from the center camera. The angle of the corresponding image from the left camera can be obtained by adding a correction (0.15) to this center angle and for the corresponding image from the right camera by subtracting this correction from the center angle (model.py lines 46-63).
These images show corresponding images from the left, center and right camera:

![alt text][image6]
![alt text][image7]
![alt text][image8]

Track one has a left turn bias. To generalize the model better, I augmented the data set by flipping the images and taking the opposite sign of their corresponding steering angles (model.py lines 65-74). For example, here is an image that has then been flipped:

![alt text][image9]
![alt text][image10]

I randomly shuffled the data set (model.py lines 30, 44, 85) and put 20% of the data into a validation set (model.py line 88).

###Simulation

####Ability to navigate correctly on test data

The vehicle is able to drive autonomously around the track without leaving the road:

![alt text][video1]
