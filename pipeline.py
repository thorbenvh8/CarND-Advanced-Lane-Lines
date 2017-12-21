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

def color_gradient_frames(frames, s_thresh=(125, 255), sx_thresh=(10, 100), sobel_kernel = 3):
    import cv2
    import numpy as np
    from util import printProgressBar

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

        # Sobelx - takes the derivate in x, absolute value, then rescale
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobelx)
        sxbinary[(scaled_sobelx >= sx_thresh[0]) & (scaled_sobelx <= sx_thresh[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        # If two of the three are activated, activate in the binary image
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(sxbinary == 1) | (s_binary == 1)] = 1

        color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
        color_binaries.append(color_binary)
        # Update Progress Bar
        printProgressBar(i+1, len_frames, 'Pipeline frames')
    return color_binaries

def get_perspective_transform(height = 720, width = 1280):
    import numpy as np
    import cv2

    # Source points - defined area of lane line edges
    src_offset_width_bottom = 235 - 300
    src_offset_width_top = 45 + 30
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

    # Use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(np.float32(src), dst)
    return M, width, height, src

def birds_eye_frames(frames, M, width, height, src):
    import cv2
    import numpy as np
    from util import printProgressBar

    birds_eye_frames = []
    len_frames = len(frames)
    # Initial call to print 0% progress
    printProgressBar(0, len_frames, 'Birds eye frames')

    for i in range(len_frames):
        frame = frames[i]

        # Draw red rectangle on frame to show src for transformation
        pts = np.array(src, np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(frame,[pts],True,(255,0,0))
        birds_eye_frames.append(frame)

        # Use cv2.warpPerspective() to warp the image to a top-down view
        birds_eye_frame = cv2.warpPerspective(frame, M, (width, height))
        birds_eye_frames.append(birds_eye_frame)

        # Update Progress Bar
        printProgressBar(i+1, len_frames, 'Birds eye frames')

    return birds_eye_frames

def save_frames_to_video(frames, fps, output_path = "output.mp4"):
    from moviepy.editor import ImageSequenceClip
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_path, audio=False)

mtx, dist = calc_distortion(debug = False)
frames, fps = load_frames("project_video.mp4", end_frame = 25 * 2)
#frames, fps = load_frames("project_video.mp4", start_frame = 25 * 15, end_frame = 25 * 17)
undistorted_frames = undistort_frames(frames)
clr_gradient_frames = color_gradient_frames(undistorted_frames)
M, width, height, src = get_perspective_transform()
brd_eye_frames = birds_eye_frames(clr_gradient_frames, M, width, height, src)

save_frames_to_video(brd_eye_frames, fps, "output.mp4")
