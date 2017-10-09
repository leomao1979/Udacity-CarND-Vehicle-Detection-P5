import numpy as np

# Define a class to receive the characteristics of each line detection
class Line():
    MINIMUM_PIXEL_COUNT = 500

    def __init__(self):
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients of the last n iterations
        self.recent_fits = []
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # the count of successive detection failures
        self.successive_failure_count = 0

        # line detected but not confirmed
        self.temp_line = None
        # the threshold of successive detection failures. If exceeds, reset and start searching from scratch
        self.successive_failure_threshold = 3
        # the number of *n* fits / iterations to be used for self.bestx and self.best_fit
        self.recent_n = 5
        self.window_height = 80
        self.margin = 100
        self.minpix = 50            # minimum pixels to recenter window
        self.ym_per_pix = 30 / 720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 700 # meters per pixel in x dimension

    def detect(self, warped, isLeft):
        self.detected = False
        self.temp_line = None
        allx, ally = self._get_line_pixels(warped, isLeft)
        if len(allx) > Line.MINIMUM_PIXEL_COUNT:
            height = warped.shape[0]
            ploty = np.linspace(0, height-1, height)
            current_fit = np.polyfit(ally, allx, 2)
            x_fitted = current_fit[0]*ploty**2 + current_fit[1]*ploty + current_fit[2]
            x_fitted[x_fitted < 0] = 0
            x_fitted[x_fitted >= warped.shape[1]] = warped.shape[1]-1
            radius_of_curvature = self._measure_curvature(ploty, x_fitted, height)

            if self._sanity_check(current_fit, radius_of_curvature):
                self.detected = True
                self.temp_line = Line()
                self.temp_line.allx = allx
                self.temp_line.ally = ally
                self.temp_line.current_fit = current_fit
                self.temp_line.x_fitted = x_fitted
                self.temp_line.line_base_pos = abs(self.temp_line.x_fitted[height-1] - warped.shape[1]/2) * self.xm_per_pix
                self.temp_line.radius_of_curvature = radius_of_curvature

            self.radius_of_curvature = radius_of_curvature

        if not self.detected:
            self.successive_failure_count += 1
            print('Warning: insufficent pixels detected. isLeft: {}, failures: {}'.format(isLeft, self.successive_failure_count))

        return self.temp_line

    def _get_line_pixels(self, warped, isLeft):
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        pixel_indices = []
        if self.best_fit is None or self.successive_failure_count > self.successive_failure_threshold:
            pixel_indices = self._detect_line_pixels(warped, isLeft)
        else:
            # Skip sliding windows to search in a margin around the previous line position
            x_fitted = self.best_fit[0]*(nonzeroy**2) + self.best_fit[1]*nonzeroy + self.best_fit[2]
            pixel_indices = ((nonzerox > (x_fitted - self.margin)) & (nonzerox < (x_fitted + self.margin)))

        allx = nonzerox[pixel_indices]
        ally = nonzeroy[pixel_indices]
        return (allx, ally)

    # Detect line pixels with sliding windows
    def _detect_line_pixels(self, warped, isLeft):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(warped[int(warped.shape[0]/2):,:], axis=0)
        # Find the peak of the bottom half of the histogram
        # It will be the starting point to detect line pixels
        midpoint = np.int(histogram.shape[0]/2)
        x_base = np.argmax(histogram[:midpoint])
        if not isLeft:
            x_base = np.argmax(histogram[midpoint:]) + midpoint
        # Current positions to be updated for each window
        x_current = x_base
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Create empty list to receive lane pixel indices
        pixel_indices = []
        # Choose the number of sliding windows
        nwindows = np.int(np.ceil(warped.shape[0]/self.window_height))
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = warped.shape[0] - (window+1)*self.window_height
            win_y_high = warped.shape[0] - window*self.window_height
            win_x_low = x_current - self.margin
            win_x_high = x_current + self.margin

            # Identify the nonzero pixels in x and y within the window
            good_indices = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                            (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]
            # Append these indices to the lists
            pixel_indices.append(good_indices)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_indices) > self.minpix:
                x_current = np.int(np.mean(nonzerox[good_indices]))

        # Concatenate the array of indices
        pixel_indices = np.concatenate(pixel_indices)
        # print('Detect line pixels with sliding windows. pixels detected: {}'.format(len(pixel_indices)))
        return pixel_indices

    def _sanity_check(self, current_fit, radius_of_curvature):
        result = True
        # TBD: find a better algorthim for sanity check. radius_of_curvature is not reliable
        # if self.radius_of_curvature is not None:
        #     radius_changed = abs(radius_of_curvature/self.radius_of_curvature - 1)
        #     result = (radius_changed <= 0.5)
        #     if not result:
        #         print('radius changed: {}. old: {}, new: {}'.format(radius_changed, self.radius_of_curvature, radius_of_curvature))
        return result

    def _measure_curvature(self, ploty, x_fitted, height):
        # Fit new polynomials to x,y in world space
        x_fit_cr = np.polyfit(ploty*self.ym_per_pix, x_fitted*self.xm_per_pix, 2)
        # Calculate the new radii of curvature
        radius_of_curvature = ((1 + (2*x_fit_cr[0]*height*self.ym_per_pix + x_fit_cr[1])**2)**1.5) / np.absolute(2*x_fit_cr[0])
        # Now our radius of curvature is in meters
        return radius_of_curvature

    def detection_confirmed(self, is_valid):
        if is_valid:
            self.allx = self.temp_line.allx
            self.ally = self.temp_line.ally
            self.recent_fits.append(self.temp_line.current_fit)
            if len(self.recent_fits) > self.recent_n:
                self.recent_fits.pop(0)
            self.best_fit = np.average(self.recent_fits, axis=0)
            self.recent_xfitted.append(self.temp_line.x_fitted)
            if len(self.recent_xfitted) > self.recent_n:
                self.recent_xfitted.pop(0)
            self.bestx = np.average(self.recent_xfitted, axis=0)
            self.line_base_pos = self.temp_line.line_base_pos
            if self.successive_failure_count > 0:
                self.successive_failure_count = 0
        else:
            self.successive_failure_count += 1
