# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to detect vehicles in a video (start with the test_video.mp4 and later implement on full project_video.mp4), but the main output or product we want you to create is a detailed writeup of the project.  

**Vehicle Detection Project**
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

[//]: # (Image References)
[train_images]: output_images/train_images.jpg
[HOG_examples]: output_images/HOG_examples.jpg
[sliding_windows]: output_images/sliding_windows.jpg
[detected_test1]: output_images/detected_test1.jpg
[detected_test3]: output_images/detected_test3.jpg
[detected_test4]: output_images/detected_test4.jpg
[bounding_boxes]: output_images/bounding_boxes.jpg
[labels_map]: output_images/labels_map.jpg
[video]: output_videos/detect_project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in function `get_hog_features()` of class `VehicleClassifier` (lines 66 through 79 of the file called `vehicleclassifier.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Training Data Examples][train_images]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![HOG Examples][HOG_examples]

#### 2. Explain how you settled on your final choice of HOG parameters.

The final choice of HOG parameters is the result of comparison of training efficiency and accuracy, false positives and detection performance.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using spatially binned color, histogram of color and YCrCb 3-channel HOG features in function `train()` of class `VehicleClassifier` (line 136 through 200 of the file `vehicleclassifier.py`).

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I tried different combinations of (ystart, ystop) and scale. Below is my final choice with the consideration of least false positives and optimized detection performance.

| Y Start  | Y Stop   |   Scale |
|:--------:|:--------:|--------:|
| 400      | 580      | 1.2     |
| 420      | 660      | 1.5     |
| 450      | 680      | 1.8     |

![Sliding Windows][sliding_windows]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images (the sub window at right-top corner is the heat map):

![Detected Vehicles][detected_test1]

![Detected Vehicles][detected_test3]

![Detected Vehicles][detected_test4]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
I combined the vehicle detection with last lane detection. Here's a [video](output_videos/detect_project_video.mp4).
The sub window at right-top corner is the heat map.

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is a frame of bounding box and its corresponding heatmap:

Bounding Boxes:

![Bounding Boxes][bounding_boxes]

Vehicle detected and heat map:

![Vehicle Detected][detected_test1]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap:

![Labels][labels_map]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The bounding boxes of detected vehicles are a little jumpy. Need to apply last N frames detection results with current one to avoid the problem. It will also help reducing the false positives.
