def calc_distortion(camera_cal_img_path = 'camera_cal', camera_cal_img_file_type = 'jpg', nx = 9, ny = 6, debug = False, write_undistorted_output = False):
    import numpy as np
    import glob
    import cv2
    from util import printProgressBar

    # Storing image path and image in a map together
    imagesMap = {}

    # Storing object points and image points
    obj_points = []
    img_points = []

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    img_paths = glob.glob(camera_cal_img_path + '/*.' + camera_cal_img_file_type)
    len_img_paths = len(img_paths)

    printProgressBar(0, len_img_paths, 'Calculate camera distortion')
    # Looping over each image path in provided path for camera calculation images
    for i in range(len_img_paths):
        img_path = img_paths[i]
        if (debug):
            print("Reading chessboard image: {}".format(img_path))
        img = cv2.imread(img_path)
        imagesMap[img_path] = img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret == True:
            if (debug):
                print("Found chessboard corners for {}".format(img_path))
            obj_points.append(objp)
            img_points.append(corners)
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        printProgressBar(i+1, len_img_paths, 'Calculate camera distortion')

    # Camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1],None,None)

    # Output undistorted images to output_images folder
    if (write_undistorted_output):
        for img_path, img in imagesMap.items():
            new_img_path = img_path.replace(camera_cal_img_path, 'output_images')
            if (debug):
                print("Writing undistorted chessboard image: {}".format(new_img_path))
            undst_img = cv2.undistort(img, mtx, dist, None, mtx)
            cv2.imwrite(new_img_path, undst_img)

    return mtx, dist

def load_frames(video_path = "project_video.mp4", start_frame = None, end_frame = None):
    from moviepy.editor import VideoFileClip
    import cv2
    from util import printProgressBar

    # The file referenced in clip1 is the original video before anything has been done to it
    input = VideoFileClip(video_path)
    #vid_clip = input.fl_image(process_image)

    len_frames = int(input.fps * input.duration)
    len_frames = len_frames if end_frame == None or end_frame > len_frames else end_frame
    i = 0
    # Initial call to print 0% progress
    printProgressBar(0, len_frames, 'Loading frames')
    frames = []
    for frame in input.iter_frames():
        if start_frame == None or i > start_frame:
            frames.append(frame)
            # Update Progress Bar
            printProgressBar(i+1, len_frames, 'Loading frames')
            if i - 1 >= len_frames:
                break
        i = i+1

    return frames, input.fps

def undistort_frames(frames):
    import cv2
    from util import printProgressBar

    undistorted_frames = []
    len_frames = len(frames)
    # Initial call to print 0% progress
    printProgressBar(0, len_frames, 'Undistort frames')
    for i in range(len_frames):
        frame = frames[i]
        undistorted_frames.append(cv2.undistort(frame, mtx, dist, None, mtx))
        # Update Progress Bar
        printProgressBar(i+1, len_frames, 'Undistort frames')
    return undistorted_frames

def color_gradient_frames(frames, s_thresh=(125, 255), sx_thresh=(30, 100), sobel_kernel = 3):
    import cv2
    import numpy as np
    from util import printProgressBar

    combined_binaries = []
    color_binaries = []
    len_frames = len(frames)
    # Initial call to print 0% progress
    printProgressBar(0, len_frames, 'Pipeline frames')
    for i in range(len_frames):
        img = np.copy(frames[i])

        # Convert to HLS colorspace
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]

        # Find the dark colors in the image to remove them from s and sx
        lower = np.array([0,0,0])
        upper = np.array([50,50,100])
        mask = cv2.inRange(img, lower, upper)
        black_binary = np.zeros_like(s_channel)
        black_binary[(mask > 0)] = 1

        # Sobelx - takes the derivate in x, absolute value, then rescale
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobelx)
        sxbinary[(black_binary == 0) & (scaled_sobelx >= sx_thresh[0]) & (scaled_sobelx <= sx_thresh[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(black_binary == 0) & (s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        # If two of the three are activated, activate in the binary image
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(sxbinary == 1) | (s_binary == 1)] = 1
        combined_binaries.append(combined_binary)

        color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
        color_binaries.append(color_binary)
        # Update Progress Bar
        printProgressBar(i+1, len_frames, 'Pipeline frames')
    return combined_binaries, color_binaries

def get_perspective_transform(height = 720, width = 1280):
    import numpy as np
    import cv2

    # Source points - defined area of lane line edges
    src = np.float32([[545, 460],
                      [735, 460],
                      [1280, 700],
                      [0, 700]])

    # 4 destination points to transfer
    dst = np.float32([[0, 0],
                      [1280, 0],
                      [1280, 720],
                      [0, 720]])

    # Use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(np.float32(src), dst)
    return M, width, height, src

def birds_eye_frames(bin_frames, M, width, height, src):
    import cv2
    import numpy as np
    from util import printProgressBar

    birds_eye_bin_frames = []
    #birds_eye_frames = []
    len_frames = len(frames)
    # Initial call to print 0% progress
    printProgressBar(0, len_frames, 'Birds eye frames')

    for i in range(len_frames):
        bin_frame = bin_frames[i]
        ### outcommented but used for debugging
        #frame = frames[i]

        # Draw red rectangle on frame to show src for transformation
        #pts = np.array(src, np.int32)
        #pts = pts.reshape((-1,1,2))
        #cv2.polylines(frame,[pts],True,(255,0,0))
        #birds_eye_frames.append(frame)

        # Use cv2.warpPerspective() to warp the image to a top-down view
        birds_eye_bin_frame = cv2.warpPerspective(bin_frame, M, (width, height))
        birds_eye_bin_frames.append(birds_eye_bin_frame)
        #birds_eye_frame = cv2.warpPerspective(frame, M, (width, height))
        #birds_eye_frames.append(birds_eye_frame)

        # Update Progress Bar
        printProgressBar(i+1, len_frames, 'Birds eye frames')

    return birds_eye_bin_frames#, birds_eye_frames

def find_draw_lanes(bin_frames, frames, M):
    from util import printProgressBar
    import numpy as np
    import cv2
    from line import Line

    found_lanes_frames = []
    drawed_lanes_frames = []

    # Left and right line
    left_line = Line()
    right_line = Line()

    # Inverted M to transform image backwards
    Minv = np.linalg.inv(M)

    len_bin_frames = len(bin_frames)
    # Initial call to print 0% progress
    printProgressBar(0, len_bin_frames, 'Finding & drawing lane frames')

    for i in range(len_bin_frames):
        bin_frame = bin_frames[i]
        frame = frames[i]

        # Assuming you have created a warped binary image called "bin_frame"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(bin_frame[int(bin_frame.shape[0]/2):,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((bin_frame, bin_frame, bin_frame))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]*3/4)
        left_line.all_x_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(bin_frame.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = bin_frame.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        left_line.all_x_current = left_line.all_x_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = bin_frame.shape[0] - (window+1)*window_height
            win_y_high = bin_frame.shape[0] - window*window_height
            win_xleft_low = left_line.all_x_current - margin
            win_xleft_high = left_line.all_x_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (0,255,0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                left_line.all_x_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        left_line.all_x = nonzerox[left_lane_inds]
        left_line.all_y = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(left_line.all_y, left_line.all_x, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, bin_frame.shape[0]-1, bin_frame.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        left_line.addXfitted(left_fitx)
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        right_line.addXfitted(right_fitx)

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Draw left line
        for x, y in zip(left_line.best_fit, ploty):
            y = int(y)
            x = int(x)
            if (x >= 0 and x < 1280 and y >= 0 and y < 720):
                out_img[y, x] = [255, 255, 0]

        # Draw left line
        for x, y in zip(right_line.best_fit, ploty):
            y = int(y)
            x = int(x)
            if (x >= 0 and x < 1280 and y >= 0 and y < 720):
                out_img[y, x] = [255, 255, 0]

        # Define y-value where we want radius of curvature
        y_eval = np.max(ploty)

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(left_line.all_y*ym_per_pix, left_line.all_x*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_line.radius_of_curvature = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_line.radius_of_curvature = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        found_lanes_frames.append(out_img)

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(frame).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_line.best_fit, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line.best_fit, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(warp_zero, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(warp_zero, Minv, (frame.shape[1], frame.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(frame, 1, newwarp, 0.3, 0)

        # Draw our radius of curvature is in meters on the result
        cv2.putText(result,'Left curverad:  {:5d}m'.format(np.int(left_line.radius_of_curvature)),(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,165,0),2,cv2.LINE_AA)
        cv2.putText(result,'Right curverad: {:5d}m'.format(np.int(right_line.radius_of_curvature)),(10,70), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,165,0),2,cv2.LINE_AA)
        cv2.putText(result,'Diff. curverad: {:5d}m'.format(np.abs(np.int(left_line.radius_of_curvature) - np.int(right_line.radius_of_curvature))),(10,110), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,165,0),2,cv2.LINE_AA)

        height = frame.shape[0]
        width = frame.shape[1]
        car_position = width/2
        l_fit_x_int = left_fit[0]*height**2 + left_fit[1]*height + left_fit[2]
        r_fit_x_int = right_fit[0]*height**2 + right_fit[1]*height + right_fit[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) /2
        center_dist = (car_position - lane_center_position) * xm_per_pix
        if center_dist < 0:
            cv2.putText(result,'Vehicle is {0:.2f}m left of center'.format(np.abs(center_dist)),(10,150), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,165,0),2,cv2.LINE_AA)
        if center_dist > 0:
            cv2.putText(result,'Vehicle is {0:.2f}m right of center'.format(np.abs(center_dist)),(10,150), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,165,0),2,cv2.LINE_AA)

        drawed_lanes_frames.append(result)

        # Update Progress Bar
        printProgressBar(i+1, len_bin_frames, 'Finding & drawing lane frames')

    return found_lanes_frames, drawed_lanes_frames

def save_frames_to_video(frames, fps, output_path = "output.mp4"):
    from moviepy.editor import ImageSequenceClip
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_path, audio=False)

mtx, dist = calc_distortion(debug = False)
frames, fps = load_frames("project_video.mp4")
#frames, fps = load_frames("project_video.mp4", end_frame = 25 * 2)
#frames, fps = load_frames("project_video.mp4", start_frame = 25 * 24, end_frame = 25 * 25)
undistorted_frames = undistort_frames(frames)
clr_gradient_bin_frames, clr_gradient_frames = color_gradient_frames(undistorted_frames)
M, width, height, src = get_perspective_transform()
brd_eye_bin_frames = birds_eye_frames(clr_gradient_bin_frames, M, width, height, src)
found_lanes_frames, drawed_lanes_frames = find_draw_lanes(brd_eye_bin_frames, frames, M)
save_frames_to_video(drawed_lanes_frames, fps, "project_video_output.mp4")
