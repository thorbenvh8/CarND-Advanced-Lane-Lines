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
    i = 0 if start_frame == None else start_frame
    # Initial call to print 0% progress
    printProgressBar(0, len_frames, 'Loading frames')
    frames = []
    for frame in input.iter_frames():
        frames.append(frame)
        i = i+1
        # Update Progress Bar
        printProgressBar(i, len_frames, 'Loading frames')
        if len_frames <= i:
            break
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

def save_frames_to_video(frames, fps, output_path = "output.mp4"):
    from moviepy.editor import ImageSequenceClip
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_path, audio=False)

mtx, dist = calc_distortion(debug = False)
frames, fps = load_frames("project_video.mp4", end_frame = 50)
undistored_frames = undistort_frames(frames)
save_frames_to_video(frames, fps, "output.mp4")
