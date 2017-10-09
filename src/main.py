import cv2
import os
import numpy as np
import time
import glob
import random
from matplotlib import pyplot as plt
from matplotlib import image  as mpimg
from moviepy.editor import VideoFileClip
from lanedetector import LaneDetector
from vehicleclassifier import VehicleClassifier

def undistort_image(fullname):
    filename = fullname.split('/')[-1]
    img = mpimg.imread(fullname)
    undistorted = detector.cameraCalibrator.undistort(img)
    # print('Camera Matrix: \n{}'.format(detector.cameraCalibrator.cameraMatrix))
    # print('Distortion Coefficients: \n{}'.format(detector.cameraCalibrator.distortionCoeffs))
    save_as = output_dir + 'undistort_' + filename
    imagedata_list = [(img, 'Original Image', None), (undistorted, 'Undistorted Image', None)]
    show_images(imagedata_list, save_as=save_as)

# imagedata_list: [(image, title, cmap), ...]
def show_images(imagedata_list, save_as=None, fontsize=12):
    ncols = 3
    if len(imagedata_list) <= 3:
        ncols = len(imagedata_list)
        nrows = 1
    else:
        nrows = int(np.ceil(len(imagedata_list)/3))

    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    f, axes = plt.subplots(nrows, ncols, squeeze=False)
    f.tight_layout()
    f.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.)
    # f.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0., hspace=-0.5)
    # if nrows == 1:
    #     plt.setp(axes, xticks=np.arange(0, 1280, 250))

    axes = np.reshape(axes, -1)
    for imagedata, ax in zip(imagedata_list, axes):
        (img, title, cmap) = imagedata
        if cmap is not None:
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=fontsize)

    extra_axes_count = len(axes) - len(imagedata_list)
    for i in range(extra_axes_count):
        plt.delaxes(axes[len(imagedata_list)+i])

    if save_as is not None:
        plt.savefig(save_as, bbox_inches='tight', dpi=300)
    plt.show()

def draw_lines(img, points, top_line = True):
    line_img = np.zeros_like(img, dtype=np.uint8)
    if top_line:
        cv2.polylines(line_img, np.int32([points]), isClosed=False, color=[255, 0, 0], thickness=5)
    else:
        cv2.line(line_img, tuple(points[0]), tuple(points[1]), color=[255, 0, 0], thickness=5)
        cv2.line(line_img, tuple(points[2]), tuple(points[3]), color=[255, 0, 0], thickness=5)
    return cv2.addWeighted(img, 0.8, line_img, 1.0, 0.)

def warp_images(fullname):
    filename = fullname.split('/')[-1]
    img = mpimg.imread(fullname)
    img = detector.cameraCalibrator.undistort(img)
    warped = detector.perspectiveTransformer.warp_perspective(img)
    img = draw_lines(img, detector.perspectiveTransformer.srcPoints, top_line=True)
    warped = draw_lines(warped, detector.perspectiveTransformer.dstPoints, top_line=False)
    # print('srcPoints: {}'.format(np.int32(detector.perspectiveTransformer.srcPoints)))
    # print('dstPoints: {}'.format(np.int32(detector.perspectiveTransformer.dstPoints)))
    save_as = output_dir + 'warped_' + filename
    imagedata_list = [(img, 'Undistorted Image', None), (warped, 'Warped Image', None)]
    show_images(imagedata_list, save_as=save_as)

def sobelx_image(img, threshold):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    binary = np.zeros_like(scaled_sobelx)
    binary[(scaled_sobelx >= threshold[0]) & (scaled_sobelx <= threshold[1])] = 1
    return binary

def generate_binary_image(fullname):
    filename = fullname.split('/')[-1]
    img = mpimg.imread(fullname)
    img = detector.cameraCalibrator.undistort(img)

    imagedata_list = [(img, 'Undistorted Image', None)]
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float)
    # imagedata_list.append((gray, 'Gray Image', 'gray'))
    # gray_x_binary = sobelx_image(gray, (20, 100))
    # imagedata_list.append((gray_x_binary, 'Gradient X (Gray Image)', 'gray'))

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    hls_h_binary = sobelx_image(h_channel, (15, 100))
    imagedata_list.append((hls_h_binary, 'Gradient X (HLS H-Channel)', 'gray'))
    # imagedata_list.append((l_channel, 'HLS L-Channel', 'gray'))
    hls_l_binary = sobelx_image(l_channel, (30, 90))
    imagedata_list.append((hls_l_binary, 'Gradient X (HLS L-Channel)', 'gray'))
    # imagedata_list.append((s_channel, 'HLS S-Channel', 'gray'))
    # Threshold color channel
    s_thresh = (90, 255)
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    imagedata_list.append((s_binary, 'Color Threshold (HLS S-Channel)', 'gray'))

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
    hsv_h_channel = hsv[:,:,0]
    hsv_s_channel = hsv[:,:,1]
    hsv_v_channel = hsv[:,:,2]
    # hsv_h_binary = sobelx_image(hsv_h_channel, (15, 100))
    # imagedata_list.append((hsv_h_binary, 'Gradient X (HSV H-Channel)', 'gray'))
    # imagedata_list.append((hsv_s_channel, 'HSV S-Channel', 'gray'))
    # imagedata_list.append((hsv_v_channel, 'HSV V-Channel', 'gray'))
    hsv_s_binary = np.zeros_like(hsv_s_channel)
    hsv_s_binary[(hsv_s_channel >= s_thresh[0]) & (hsv_s_channel <= s_thresh[1])] = 1
    imagedata_list.append((hsv_s_binary, 'Color Threshold (HSV S-Channel)', 'gray'))

    combined_binary = detector.generate_binary_image(img)
    imagedata_list.append((combined_binary, 'HSV for Yellow & RGB for White', 'gray'))
    save_as = output_dir + 'binary_' + filename
    show_images(imagedata_list, save_as, fontsize=7)

def detect_lane(fullname):
    filename = fullname.split('/')[-1]
    img = mpimg.imread(fullname)
    detected = detector.detect_lane(img)
    save_as = output_dir + 'detect_lane_' + filename
    imagedata_list = [(img, 'Original Image', None), (detected, 'Lane Detected Image', None)]
    show_images(imagedata_list, save_as=save_as)

def process_image(img):
    return detector.detect_lane(img)

def detect_lane_for_video():
    # video_file = 'project_video.mp4'
    video_file = '../test_video.mp4'
    # video_file = 'challenge_video.mp4'
    # video_file = 'harder_challenge_video.mp4'
    # clip = VideoFileClip(video_file).subclip(38, 43)
    clip = VideoFileClip(video_file)
    new_clip = clip.fl_image(process_image)
    video_output = 'output_videos/detect_' + video_file
    new_clip.write_videofile(video_output, audio=False)

def detect_vehicles(fullname):
    filename = fullname.split('/')[-1]
    img = mpimg.imread(fullname)
    window_img, label_img = classifier.detect_vehicles(img)
    plt.imshow(window_img)
    plt.show()
    plt.imshow(label_img)
    plt.show()

def process_image_vehicle(img):
    img = detector.detect_lane(img)
    _, result = classifier.detect_vehicles(img)
    return result

def detect_vehicles_for_video():
    # video_file = 'test_video.mp4'
    video_file = 'project_video.mp4'
    clip = VideoFileClip('../' + video_file)
    # clip = VideoFileClip('../' + video_file).subclip(10,20)
    new_clip = clip.fl_image(process_image_vehicle)
    video_output = '../output_videos/detect_' + video_file
    new_clip.write_videofile(video_output, audio=False)

detector = LaneDetector()
classifier = VehicleClassifier()
# detect_vehicles_for_video()

# detect_vehicles('../test_images/test1.jpg')
# detect_vehicles('../test_images/test2.jpg')
# detect_vehicles('../test_images/test3.jpg')
# detect_vehicles('../test_images/test4.jpg')
# detect_vehicles('../test_images/test5.jpg')
# image_dir = 'test_images/'
# output_dir = 'output_images/'
# detect_lane_for_video()

# undistort_image('camera_cal/calibration1.jpg')
# undistort_image('test_images/test1.jpg')
# warp_images('test_images/straight_lines1.jpg')
# generate_binary_image('test_images/test1.jpg')
# generate_binary_image('test_images/challenge1.jpg')
# detect_lane('test_images/challenge1.jpg')

def load_random_images():
    images = glob.glob('../train_data/**/*.png', recursive=True)
    cars = []
    notcars = []
    for image in images:
        if 'non-vehicles' in image:
            notcars.append(image)
        else:
            cars.append(image)
    randint = random.randint(0, len(cars))
    car_img = mpimg.imread(cars[randint])
    randint = random.randint(0, len(notcars))
    notcar_img = mpimg.imread(notcars[randint])
    imagedata_list = [(car_img, 'Car', None), (notcar_img, 'Not-Car', None)]
    show_images(imagedata_list, save_as='../output_images/train_images.jpg')

load_random_images()
