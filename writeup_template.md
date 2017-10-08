#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/class_dist.png "Visualization"
[image2]: ./extra_data/children_crossing.png
[image3]: ./extra_data/limit_80.png
[image4]: ./extra_data/priority_road.png
[image5]: ./extra_data/keep_right.png
[image6]: ./extra_data/yield.png 
[image7]: ./extra_data/no_entry.png
[image8]: ./extra_data/limit_30.png 

[image9]: ./examples/real_images1.png
[image10]: ./examples/real_images2.png
[image11]: ./examples/right_ahead.jpeg
[image12]: ./examples/real_images3.png

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.


###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

####2. Include an exploratory visualization of the dataset.

Visualization of the data set, I just show one picture of each class. And using a bar chart to show the distribution of the train set (see below).


![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

I tried to convert the images to grayscale, but I find it reduced the recognition rate. Perhaps the color contains information to help separate different trafic signs. I use pixel / 255 - 0.5 to normalize the images, and it significally increased the recognition rate. 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers(LeNet):

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, output 10x10x16    |
| RELU  				|												|
| Max polling			| 2x2 stride, output 5x5x16				        |
| Flatten				| output 400									|
| Fully connected		| output 120									|
| RELU					|												|
| Fully connected		| output 84										|
| RELU					|												|
| Fully connected		| output 43        								|
| Softmax				| output probability     						|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

EPOCHS = 180
BATCH_SIZE = 128
learning rate = 0.001

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.00.
* validation set accuracy of 0.948.
* test set accuracy of 0.936.


If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
LeNet is used in this project. Because it is designed to recognize hand-written digits, a similar sturcture get a very good result at ImageNet competition in 2012. It is als suggessted as a start of this projcet. With simple preprocess, this method can get a result that meets the project requirement.

###Test a Model on New Images

1. I got eight German traffic sings from Berlin, where I currently live:



![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6] ![alt text][image7]
![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:


| Image			        |     Prediction	        |  		Result		    | 
|:---------------------:|:-------------------------:|:---------------------:| 
| Children Crossing		| Children Crossing		    |						| 
| 80 km/h    			| Priority Road				|			    		|
| Priority Road			| Priority Road				|					    |
| Yield     			| Yield               		|					    |
| No Entry              | Turn Right Ahead          |      Wrong            |


The model can only recognize 4 of the 5 trafic signs correctly. So the accuracy is 80%, lower than the accuracy on the test set of 93.6%. 

The following shows the wrongly classified No Entry sign, the normalized image and the softmax probability. It is recognized as Turn Right Ahead sign with strong confidence. The probability is almost 1.0. The last two images are Turn Right Ahead sign and the image after normalization, can't say these two traffic sings have much similarity, but after the simple normalization, they have similar background color. I think I will need to improve the preprocessing if I want to get a better recognition rate.


![image7]![image9]![image10]![image11]![image12]
####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


The model is quite certain about its prediction. It has aroud 1.0 probability for four of the five images.
Only for the Speed Limit (80 km/h), the top five soft max probabilities are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .91         			| Speed Limit 80 km/h   						| 
| .05     				| No passing for vehicles 3.5 metric tons		|
| .03					| Priority Road									|
| .01	      			| Speed Limit 100 km/h			 				|
| .00				    | Speed Limit 50 km/h   						|






