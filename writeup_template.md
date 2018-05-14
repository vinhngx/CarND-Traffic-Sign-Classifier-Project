# **Traffic Sign Recognition** 

---

[//]: # (Image References)

[image1]: ./web_examples/construction.jpeg "Construction"
[image2]: ./web_examples/kindergarten.jpeg "Kindergarten"
[image3]: ./web_examples/noentry.jpg "No entry"
[image4]: ./web_examples/speed_70.jpeg "Speed limit 70"
[image5]: ./web_examples/stop.jpeg "Stop"

[top-5-prob]: ./top-5-prob.png "Top 5 Probabilities"


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf.

You're reading it! and here is a link to my [project code](https://github.com/vinhngx/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set

Number of training examples = 34799

Number of validation examples = 4410

Number of testing examples = 12630

Image data shape = (32, 32)

Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

We used matplotlib library to plot class distribution amongst the three train, test and validation sets.
For each class, we select and plot a random set of 10 images.
See  within [project code](https://github.com/vinhngx/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) 

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data
A simple data standardization step is employed, where each images is scaled to the range [-1, 1]

X_train_normalized = (X_train - 128.)/128

This is a quick way to standardize the images to have approximately zero mean and unit variance distribution. Standardizing the data should help to condition the optimization problem and facilitate the optimization process. Also, it puts quantities, such as the learning rate, into a scale perspective.  

Since the model has archieved a good converged accuracy, we have not tested data augmentation herein.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x16 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x16 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x32 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x64 				|
| Global average pooling 2D | Output 64 |
| Flatten | Output 64 |
| Dropout | keep_prob = 0.5 |
| Fully connected		| output 43	|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer with L2 regularization (weight 1e-4) with learning rate 1e-3. The Adam optimizer was chosen, since it can automatically adapt the effective learning rate, and thus serves as an excellent tool for quick model exploration. 

The L2 regularization parameter was chosen empirically from 1e-2, 1e-3, 1e-4 based on the validation set. We notice larger values of the L2 penalty to be too strong and prevent the model from learning. 

We chose a batchsize of 1024 to improve GPU efficiency. GPUs are best at processing large batches of data. Larger batches also help to smooth the gradient approximation, thus a larger learning rate can potentially be applied.

The network is trained until convergence for 500 epoch. During training, train and validation loss and accuracy is monitored to ensure the traning is progressing well and no anomaly occurs. We notice that after 300 epoch, although the improvement in accuracy has slowed down, prolonging the training can still be benefical if an absolutely best model is sought after.   


Training...

EPOCH 1 Loss = 3.597 Validation loss: 3.140 Train Accuracy = 0.057 Validation Accuracy = 0.084

EPOCH 2 Loss = 3.378 Validation loss: 2.958 Train Accuracy = 0.100 Validation Accuracy = 0.111

EPOCH 3 Loss = 3.232 Validation loss: 2.876 Train Accuracy = 0.117 Validation Accuracy = 0.131

...

EPOCH 498 Loss = 0.159 Validation loss: 0.115 Train Accuracy = 0.981 Validation Accuracy = 0.976

EPOCH 499 Loss = 0.159 Validation loss: 0.117 Train Accuracy = 0.981 Validation Accuracy = 0.974

EPOCH 500 Loss = 0.158 Validation loss: 0.109 Train Accuracy = 0.980 Validation Accuracy = 0.976

Final model saved

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 98%
* validation set accuracy of 97.6% 
* test set accuracy of 97%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? 

Initially a Lenet architecture was chosens. I then modified the model using 3x3 convolution and same padding, since 5x5 filters seems large for image of size 32x32. The model was heavily overfitted, with a large gap between train and validation accuracy (99.9% and 89% respectively). To combat overfitting, I added dropout layer and employ L2 regularization. The final model parameter was chosen based on the validation set accuracy.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5]

These images are brighter than those seen in training images (which are extracted from videos).

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Road work     			| Road work 										|
| Children crossing					| Speed limit 30											|
| 70 km/h	      		| 70 km/h					 				|
| No entry		| No entry     							|

![alt text][image4]

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The top-1 and top-5 soft max probabilities are detailed in the table and figure below:

| Probability			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100%     		| Stop sign   									| 
| 100%      			| Road work 										|
| 87.29%				| Speed limit 30											|
| 100%      		| 70 km/h					 				|
| 100%	| No entry     							|

![top-5-prob][top-5-prob]

For the 3rd image, the correct label comes the 2nd highest amongst the top 5 predictions.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


