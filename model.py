import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
import cv2
import csv
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dropout, Flatten, Dense, Activation
import matplotlib.pyplot as plt

# command line flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('steering_correction', 0.15, 'Steering correction for the side camera images')
flags.DEFINE_string('training_file', './data/driving_log.csv', 'driving_log training file (.csv)')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_integer('nb_epoch', 5, 'Number of epochs')

def generator(samples, batch_size):
    """
    Loading, preprocessing and augmentation of the sample data in batches separately from the main routine
    :param samples: The sample data to be processed
    :param batch_size: The batch size of sample data to be processed
    :return: Features and labels
    """
    num_samples = len(samples)
    while 1: # loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            features, labels = [], []
            for row in batch_samples:
                for col in range(3): # first three csv columns contain image paths
                    # add center, left or right image and its corresponding steering angle
                    img, steering = read_image_and_steering(row, col, FLAGS.steering_correction)
                    features.append(np.array(img, dtype=np.float32))
                    labels.append(steering)
                    # add flipped image and its corresponding steering angle
                    img_flipped, steering_flipped = flip(img, steering)
                    features.append(np.array(img_flipped, dtype=np.float32))
                    labels.append(steering_flipped)
            yield shuffle(np.array(features), np.array(labels))

def read_image_and_steering(csv_row, csv_img_col, steering_correction):
    """
    Read in the image and adjust steering from center, left or right camera
    :param csv_row: The row from the csv file to be read
    :param csv_img_col: The column from the csv file that contains the image path to be read
    :param steering_correction: The steering correction for the side (left, right) camera images
    :return: Image and (adjusted) steering
    """
    img = cv2.imread('./data/IMG/' + csv_row[csv_img_col].split('/')[-1]) # No MS Windows
    #img = cv2.imread('./data/IMG/' + csv_row[csv_img_col].split('\\')[-1]) # MS Windows
    steering_center = float(csv_row[3])
    if csv_img_col == 0: # center image
        steering = steering_center
    if csv_img_col == 1: # left image
        steering = steering_center + steering_correction
    if csv_img_col == 2: # right image
        steering = steering_center - steering_correction
    return img, steering

def flip(img, steering):
    """
    Flip the image and steering angle around y-axis
    :param img: The image to be flipped
    :param steering: The steering angle to be flipped
    :return: Flipped image and steering
    """
    img_flipped = cv2.flip(img, 1)
    steering_flipped = -steering
    return img_flipped, steering_flipped

def main(_):
    # load the train data
    samples = []
    with open(FLAGS.training_file) as csvfile:
        driving_log = csv.reader(csvfile)
        for row in driving_log:
            samples.append(row)
    
    # shuffle the train data
    samples = shuffle(samples)
    
    # split data into training and validation sets
    samples_train, samples_val = train_test_split(samples, test_size=0.2, random_state=0)
    
    # create the Sequential model
    model = Sequential()
    
    #  lambda layer to parallelize image normalization
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    
    # cropping2D layer
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    
    # 3 convolutional layers with a 2×2 stride and a 5×5 kernel
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    
    model.add(Dropout(0.2))
    
    # 2 non-strided convolutional layers with a 3×3 kernel
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    
    model.add(Dropout(0.3))
    model.add(Flatten())
    
    # 3 fully connected layers with a rectifier activation function
    model.add(Dense(100))
    model.add(Dropout(0.4))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    
    # fully connected output layer
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    
    print(model.summary())
    
    # train the model using the generator function
    generator_train = generator(samples_train, FLAGS.batch_size)
    generator_val = generator(samples_val, FLAGS.batch_size)
    history_object = model.fit_generator(generator_train, samples_per_epoch=len(samples_train), \
        validation_data=generator_val, nb_val_samples=len(samples_val), nb_epoch=FLAGS.nb_epoch, verbose=1)
    
    model.save('./model.h5') # creates a HDF5 file 'model.h5'
    
    ### print the keys contained in the history object
    print(history_object.history.keys())
    
    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
