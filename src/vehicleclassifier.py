import glob
import time
import os
import cv2
import pickle
import random
import numpy as np
import scipy
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
        self.spatial_size=(16, 16)
        self.hist_bins=16
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

        if self.svc is None or self.X_scaler is None:
            print('self.svc is not trained. Start training ...')
            self.train()
            print('Done')
        else:
            self.isTrained = True
            print('Vehicle calssifier has been trained. Skip.')

    # Define a function to compute binned color features
    def bin_spatial(self, img):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, self.spatial_size).ravel()
        return features

    # Define a function to compute color histogram features
    def color_hist(self, img):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=self.hist_bins, range=self.bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=self.hist_bins, range=self.bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=self.hist_bins, range=self.bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
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
        features = []
        # Iterate through the list of images
        for img_file in imgs:
            image = mpimg.imread(img_file)
            if image.dtype == np.float32 or image.type == np.float64:
                # mpimg will read png files on a scale of 0 ~ 1. Change to 0 ~ 255
                image = (image * 255).astype(np.uint8)
            img_features = self.single_img_features(image)
            features.append(img_features)
        return features

    # Convert img in 'RGB' to self.color_space
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

        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        self.svc = LinearSVC()
        self.X_scaler = StandardScaler()
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
            self.isTrained = True

    def draw_boxes(self, img, bboxes, color = None, thick=6):
        imcopy = np.copy(img)
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates in random color
            if color is None:
                random_color = [random.randint(0, 255) for _ in range(3)]
                cv2.rectangle(imcopy, bbox[0], bbox[1], random_color, thick)
            else:
                cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        return imcopy

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
        allwindows = []
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
                test_prediction = self.svc.predict(test_features)

                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                allwindows.append(((xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                if test_prediction == 1:
                    windows.append(((xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart)))

        return (windows, allwindows)

    def find_cars(self, img):
        y_scales = [(400, 580, 1.2), (420, 660, 1.5), (450, 680, 1.8)]
        windows = []
        all_windows = []
        for ystart, ystop, scale in y_scales:
            scale_windows, scale_allwindows = self.find_cars_with_scale(img, ystart, ystop, scale)
            windows.extend(scale_windows)
            all_windows.extend(scale_allwindows)
        return (windows, all_windows)

    def draw_labeled_bboxes(self, img, labels):
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0, 255, 0), 6)
        return img

    def build_heatmap(self, image, bbox_list):
        heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        # Apply threshold to help remove false positives
        heatmap[heatmap < self.HEAT_THRESHOLD] = 0
        # Visualize the heatmap when displaying
        heatmap = np.clip(heatmap, 0, 255)

        return heatmap

    def _add_sub_window(self, image, heat):
        top_gap = 5
        gap_between_windows = 8
        sub_window_size = (np.int_(image.shape[0]/3), np.int_((image.shape[1]-gap_between_windows*4)/3))

        # Heat image sub window
        max_heat = np.max(heat)
        if max_heat > 0:
            heat = (heat/max_heat * 255).astype(np.int)
        heat_image = np.dstack((heat, np.zeros_like(heat), np.zeros_like(heat)))
        heat_image = scipy.misc.imresize(heat_image, sub_window_size)
        image[top_gap:sub_window_size[0]+top_gap, -gap_between_windows-sub_window_size[1]:-gap_between_windows] = heat_image
        return image

    def detect_vehicles(self, image):
        hot_windows, all_windows = self.find_cars(image)
        # for window in hot_windows:
        #     print('window: {}'.format(window))
        # window_img = self.draw_boxes(image, all_windows, thick=2)
        window_img = self.draw_boxes(image, hot_windows, thick=6)
        heatmap = self.build_heatmap(image, hot_windows)
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        label_img = self.draw_labeled_bboxes(np.copy(image), labels)
        label_img = self._add_sub_window(label_img, heatmap)

        return (window_img, label_img)
