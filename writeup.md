## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2a]: ./writeup/original.png "Road Original"
[image2b]: ./writeup/transformed.png "Road Transformed"
[image3]: ./writeup/binary_combo.png "Binary Example"
[image4a]: ./writeup/src_warped_curve.png "Warp Example Source"
[image4b]: ./writeup/dst_warped_curve.png "Warp Example Destination"
[image5a]: ./writeup/histogram.png "Histogram"
[image5b]: ./writeup/color_fit_lines.png "Fit Visual"
[image6]: ./writeup/output.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is in the function `calc_distortion` in the lines #1 through #51 in the code of the Python file located in `./pipeline.py` to calculate the distortion based on the images in "./camera_cal".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]
or look into the folder "./output_images" to see a transformation of each image from "./camera_cal".

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

And undistorted image from the video looks like this:
![alt text][image2a]
![alt text][image2b]

I do this in the function `undistort_frames` to each image of the video in line #79 through #92 in `pipeline.py`.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in function `color_gradient_frames` at lines #94 through #134 in `pipeline.py`).  Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `get_perspective_transform()`, which appears in lines #136 through #157 in the file `pipeline.py`.  The `get_perspective_transform()` calculates me a `M`for a specific `height` and `width`.  I chose the hardcode the source and destination points in the following manner:

```python
  # Source points - defined area of lane line edges
  src_offset_width_bottom = -65
  src_offset_width_top = 75
  src_offset_height = 90
  bottom_left = [src_offset_width_bottom, height]
  bottom_right = [width - src_offset_width_bottom + 100, height]
  top_right = [width / 2 + src_offset_width_top, height / 2 + src_offset_height]
  top_left = [width / 2 - src_offset_width_top, height / 2 + src_offset_height]
  src = [bottom_left, bottom_right, top_right, top_left]

  # 4 destination points to transfer
  offset = 300 # offset for dst points
  dst = np.float32([[offset, height],[width-offset, height],
                    [width-offset, 0],[offset, 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| -65, 720      | 300, 720      |
| 1445, 720     | 980, 720      |
| 715, 450      | 980, 0        |
| 565, 450      | 300, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Source image][image4a]
![Destination (warped) image][image4b]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did an histogram over the warped image to check for the sum from the bottom to the half of the screen. So from the closest to the car to half through the image height. The maximums of the histogram could be a potential line. We only search from the half and the offset we set when we warped the image.

![alt text][image5a]

From this starting point we create windows (in my case 9) to search for the line in there and then move further up.
We identify the nonzero pixels in the window and if we found more on the left side then the right side of the window we shift the next window on top to the mean of pixels. So the window moves to the left or to the right when its a curve etc to identify the line.
From the identified line we calculate the curvade of each left and right lane.

You find this code in function `find_draw_lanes` at lines #195 through #307 of `./pipeline.py`.

![alt text][image5b]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I assume in my code the following ratio from pixel to meter:
```python
  # Define conversions in x and y from pixels space to meters
  ym_per_pix = 30/720 # meters per pixel in y dimension
  xm_per_pix = 3.7/700 # meters per pixel in x dimension
```
For the curverad calculation you find this code in function `find_draw_lanes` at lines #309 through #321 of `./pipeline.py`.

For the center of the car calculation you find this code in function `find_draw_lanes` at lines #344 through #354x of `./pipeline.py`.

To make a smother lane we take the mean of the last n found lines and calculate a mean over it when we add the newest line. Checkout code from `pipeline.py` at line #289 through #293 and in `line.py` at line #29 through #34. Then we use this calculated mean from the class `Line` as `best_fit` in `pipeline.py` at line #327 through #328.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The most important step is to draw the lane and then warp it into the perspective of the original. We achieve this by inverting the `M` we calculated before and then use `cv2.warpPerspective`. Then we use a little weighted merging to have a transparent effect of our line drawn on the original.

I implemented this step in lines #325 through #345 in my code in `pipeline.py` in the function `find_draw_lanes()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My pipeline would currently fail if it doesn't detect a line on the image provided. If I can't find a lane, I could use the old detected line.
