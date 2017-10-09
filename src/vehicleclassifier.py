import glob
import time
import os
import cv2
import pickle
import random
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from skimage.feature import hog
from scipy.ndimage.measurements import label

class VehicleClassifier:
    HEAT_THRESHOLD = 2

    def __init__(self):
        self.color_space='YCrCb'      # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.spatial_size=(32, 32)
        self.hist_bins=32
        self.bins_range=(0, 256)
        self.orient=9               # HOG orientation
        self.pix_per_cell=8
        self.cell_per_block=2
        self.hog_channel='ALL'          # Can be 0, 1, 2, or "ALL"

        self.spatial_feat=True
        self.hist_feat=True
        self.hog_feat=True

        self.svc = None
        self.X_scaler = None
        self.isTrained = False
        self.pickle_file = '../vehicle_classifier.p'

        if os.path.exists(self.pickle_file):
            with open(self.pickle_file, 'rb') as f:
                self.svc, self.X_scaler = pickle.load(f)
                # dist_pickle = pickle.load(f)
                # self.svc = dist_pickle["svc"]
                # self.X_scaler = dist_pickle["scaler"]
                # self.orient = dist_pickle["orient"]
                # self.pix_per_cell = dist_pickle["pix_per_cell"]
                # self.cell_per_block = dist_pickle["cell_per_block"]
                # self.spatial_size = dist_pickle["spatial_size"]
                # self.hist_bins = dist_pickle["hist_bins"]

        if self.svc is None or self.X_scaler is None:
            print('SVC is not trained. Start training ...')
            self.train()
            print('Done')
        else:
            self.isTrained = True
            print('Vehicle calssifier has been trained. Skip.')

    # Define a function to compute binned color features
    def bin_spatial(self, img):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, self.spatial_size).ravel()
        # Return the feature vector
        return features

    # Define a function to compute color histogram features
    # NEED TO CHANGE bins_range if reading .png files with mpimg!
    def color_hist(self, img):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=self.hist_bins, range=self.bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=self.hist_bins, range=self.bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=self.hist_bins, range=self.bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    # Define a function to return HOG features and visualization
    def get_hog_features(self, img, vis=False, feature_vec=True):
        # Call with two outputs if vis==True
        if vis == True:
            features, hog_image = hog(img, orientations=self.orient, pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                      cells_per_block=(self.cell_per_block, self.cell_per_block), transform_sqrt=True,
                                      visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:
            features = hog(img, orientations=self.orient, pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                           cells_per_block=(self.cell_per_block, self.cell_per_block), transform_sqrt=True,
                           visualise=vis, feature_vector=feature_vec)
            return features

    # Define a function to extract features from a list of images
    def extract_features(self, imgs):
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for img_file in imgs:
            # Read in each one by one
            image = mpimg.imread(img_file)
            if image.dtype == np.float32 or image.type == np.float64:
                image = (image * 255).astype(np.uint8)
            img_features = self.single_img_features(image)
            # Append the new feature vector to the features list
            features.append(img_features)
        # Return list of feature vectors
        return features

    # Convert img ('RGB') to self.color_space
    def convert_color(self, img):
        if self.color_space == 'RGB':
            feature_image = np.copy(img)
        elif self.color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif self.color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif self.color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif self.color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif self.color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

        return feature_image

    # Define a function to extract features from a single image
    def single_img_features(self, img):
        #1) Define an empty list to receive features
        img_features = []
        #2) Apply color conversion if other than 'RGB'
        feature_image = self.convert_color(img)

        #3) Compute spatial features if flag is set
        if self.spatial_feat == True:
            img_features.append(self.bin_spatial(feature_image))
        #4) Compute histogram features if flag is set
        if self.hist_feat == True:
            img_features.append(self.color_hist(feature_image))
        #5) Compute HOG features if flag is set
        if self.hog_feat == True:
            if self.hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.extend(self.get_hog_features(feature_image[:,:,channel],
                                        vis=False, feature_vec=True))
            else:
                hog_features = self.get_hog_features(feature_image[:,:,self.hog_channel], vis=False, feature_vec=True)
            img_features.append(hog_features)

        #6) Return concatenated array of features
        return np.concatenate(img_features)

    def train(self):
        if self.isTrained:
            return

        # Divide up into cars and notcars
        images = glob.glob('../train_data/**/*.png', recursive=True)
        cars = []
        notcars = []
        for image in images:
            if 'non-vehicles' in image:
                notcars.append(image)
            else:
                cars.append(image)

        # Reduce the sample size because HOG features are slow to compute
        # The quiz evaluator times out after 13s of CPU time
        # sample_size = 100
        # cars = cars[0:sample_size]
        # notcars = notcars[0:sample_size]

        t=time.time()
        car_features = self.extract_features(cars)
        notcar_features = self.extract_features(notcars)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to extract HOG features...')
        self.svc = LinearSVC()
        self.X_scaler = StandardScaler()

        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        self.X_scaler.fit(X)
        # Apply the scaler to X
        scaled_X = self.X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

        print('Using:',self.orient,'orientations',self.pix_per_cell,
            'pixels per cell and', self.cell_per_block,'cells per block')
        print('Feature vector length:', len(X_train[0]))

        # Check the training time for the SVC
        t=time.time()
        self.svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(self.svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t=time.time()
        n_predict = 10
        print('My SVC predicts: ', self.svc.predict(X_test[0:n_predict]))
        print('For these',n_predict, 'labels: ', y_test[0:n_predict])
        t2 = time.time()
        print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

        # Write to pickle file
        with open(self.pickle_file, 'wb') as f:
            pickle.dump([self.svc, self.X_scaler], f)

    # Here is your draw_boxes function from the previous exercise
    def draw_boxes(self, img, bboxes, thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            random_color = [random.randint(0, 255) for _ in range(3)]
            cv2.rectangle(imcopy, bbox[0], bbox[1], random_color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    # Define a function that takes an image,
    # start and stop positions in both x and y,
    # window size (x and y dimensions),
    # and overlap fraction (for both x and y)
    def slide_window(self, img, x_start_stop=(None, None), y_start_stop=(None, None),
                        xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
        ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
        nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
        ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    # Define a function you will pass an image
    # and the list of windows to be searched (output of slide_windows())
    def search_windows(self, img, windows):
        #1) Create an empty list to receive positive detection windows
        on_windows = []
        #2) Iterate over all windows in the list
        for window in windows:
            #3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            #4) Extract features for that window using single_img_features()
            features = self.single_img_features(test_img)
            #5) Scale extracted features to be fed to classifier
            test_features = self.X_scaler.transform(np.array(features).reshape(1, -1))
            #6) Predict using your classifier
            prediction = self.svc.predict(test_features)
            #7) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
        #8) Return windows for positive detections
        return on_windows

    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars_with_scale(self, img, ystart, ystop, scale):
        img_tosearch = img[ystart:ystop,:,:]
        ctrans_tosearch = self.convert_color(img_tosearch)
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        nyblocks = (ch1.shape[0] // self.pix_per_cell) - self.cell_per_block + 1
        nfeat_per_block = self.orient*self.cell_per_block**2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = self.get_hog_features(ch1, feature_vec=False)
        hog2 = self.get_hog_features(ch2, feature_vec=False)
        hog3 = self.get_hog_features(ch3, feature_vec=False)

        windows = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*self.pix_per_cell
                ytop = ypos*self.pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

                # Get color features
                spatial_features = self.bin_spatial(subimg)
                hist_features = self.color_hist(subimg)

                # Scale features and make a prediction
                test_features = self.X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                test_prediction = self.svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    windows.append(((xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                    # cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)

        return windows

    def find_cars(self, img):
        ystart = 400
        ystop  = 656
        scales = [1, 1.5, 2.]
        windows = []
        for scale in scales:
            windows.extend(self.find_cars_with_scale(img, ystart, ystop, scale))
        return windows

    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
        # Return the image
        return img

    def build_heatmap(self, image, bbox_list):
        heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        # Apply threshold to help remove false positives
        heatmap[heatmap < self.HEAT_THRESHOLD] = 0
        # Visualize the heatmap when displaying
        heatmap = np.clip(heatmap, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = self.draw_labeled_bboxes(np.copy(image), labels)
        return draw_img

    def detect_vehicles(self, image):
        hot_windows = self.find_cars(image)
        # y_start_stop = [int(image.shape[0]/2), None]
        # windows = self.slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
        #                     xy_window=(96, 96), xy_overlap=(0.5, 0.5))
        # hot_windows = self.search_windows(image, windows)
        # for window in hot_windows:
        #     print('window: {}'.format(window))
        window_img = self.draw_boxes(image, hot_windows, thick=6)
        label_img = self.build_heatmap(image, hot_windows)

        return (window_img, label_img)
