# Steering angle prediction from a camera image as a backup service
This repository contains a source code of research conducted at the University of Debrecen.

Folders:

## Subtask1: 

Semantic segmentation model training and calculation of correlation between emitted PWM value and detected angles on rear-view camera image
## Subtask2: 

A CNN model that is trained on PWM values, and that predicts emitted PWM signal from the rear-view camera image

## The Process of Training a Model on Subtask 1  
In Subtask 1 we used a deep-learning segmentation model to find the edges of the wheels on the rear-view camera image. Using the detected lines on the image, representing the edges, we calculated the steering angle of the front wheels.  

The process contained the following steps: 

Drive the car, and record the images of the rear-view camera, along with the emitted steering and speed PWM value. 

Annotate the images, marking the wheel edges as lines on the images  

Increase the variance of the gathered data using augmentation techniques, with a 1:10 ratio. 

Used the dataset to train a segmentation model  

### Training details 

For training, we used an RTX3090 GPU, with Keras and TensorFlow Python libraries.  

As mentioned in the main article, we used a U-Net-based CNN network to train our model for image segmentation. We used for loss function the sum of dice loss and focal loss. 

For metrics, we used the IOU Score and F1 Score, both with threshold values of 0.5. 

In the validation and test dataset, we did not apply augmentation to preserve the variance of the original dataset.  

We trained the model through 40 epochs.  

The graphs of the training process are shown below: 


![alt text](train1.png?raw=true)

For the model evaluation, the test set has been used. The evaluation results are as follows: 

Loss: 0.0036411 

mean iou_score: 0.99857 

mean f1-score: 0.99928 

## The Process of Training a Model on Subtask 2  

In Subtask 2 we do not use annotated data to train a CNN model, and we do not want to find wheel edges or axes for steering angle calculation. Instead, we use the recorded PWM value to train the model, to predict the PWM, and we transform the PWM value into steering angle.  

The process contained the following steps: 

Drive the car, and record the images of the rear-view camera, along with the emitted steering and speed PWM value.  

Transform PWM values, to eliminate the difference between the steering angle of the inner and outer steering wheel during a turn (As steering is based on Ackerman geometry).  

Normalize the PWM values to the 0-1 range, and then scale it to the range [-1,1]. 

Increase the variance of the gathered data using augmentation techniques, with a 1:10 ratio. 

Used the dataset to train a regression CNN model.  

### Training details 

For training, we used an RTX3090 GPU, with Keras and TensorFlow Python libraries.  

As mentioned in the main article, we used a VGG16-based CNN network to train our model for regression. We used for loss function Mean Squared Error. 

Layers: 


In the validation and test dataset, we did not apply augmentation to preserve the variance of the original dataset.  

The graph of the training process is shown below: 


Model evaluation has been carried out on the test dataset. The result is as follows: 

K-fold validation loss: [0.003277057083323598, 0.0035040099173784256, 0.0030176243744790554, 0.00320057203029096] 

Loss on test dataset: 0.002712803892791271 
