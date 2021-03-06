## Vehicle Detection and Tracking

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: test_videos/project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 20 through 48 of the file called `ImageClassifier.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

Vehicle Image : 
![Vehicle Image](./output_images/car_image.jpg)

Non-Vehicle Image :
![Non-Vehicle Image](./output_images/notcar_image.jpg)

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

**Vehicle Image :**

Channel 0 : ![Vehicle Image](./output_images/car_image_hog_ch0.jpg)

Channel 1 : ![Vehicle Image](./output_images/car_image_hog_ch1.jpg)

Channel 2 : ![Vehicle Image](./output_images/car_image_hog_ch2.jpg)

**Non-Vehicle Image :**

Channel 0 : ![Vehicle Image](./output_images/notcar_image_hog_ch0.jpg)

Channel 1 : ![Vehicle Image](./output_images/notcar_image_hog_ch1.jpg)

Channel 2 : ![Vehicle Image](./output_images/notcar_image_hog_ch2.jpg)


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and tried to fit the SVM classifier with different combinations and i found that these parameter values helped the SVM to perform the best classification job.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in lines 49 through 70 of the file called `ImageClassifier.py`.

I trained a linear SVM using HOG features and spatial features.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to linearly search the lower part of the image at five scales (1,1.5,2,2.5,3 ) so i can detect vehicles at different depths in the image. I applied the HOG transform on the entir image and then sub-sampled it with the windows i got from the previous step.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color in the feature vector, which provided a nice result.  Here are some example images:



![alt text](./output_images/test6_3_output.jpg)
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_videos/project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a frame of the video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the frame of the video:

### Here is a frame :

![alt text](./output_images/test6_0_undistorted.jpg)

### Here is the detected Vehicle boxs

![alt text](./output_images/test6_1_unfiltered.jpg)

### Here is it's heat map

![alt text](./output_images/test6_2_heat_map.jpg)

### Here is the output after thresholding
![alt text](./output_images/test6_3_output.jpg)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

##### To improve the detection we could do the following :

** 1 - Augment the training data and remove the time-series images from the dataset.**

** 2 - Use a more powerfull classification technique like CNN **

** 3 - To eleminate false positives from areas out of road boundary we can define the road boundaries and then limit our window search to this area. **

