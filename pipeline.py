def calc_distortion(camera_cal_img_path = 'camera_cal', camera_cal_img_file_type = 'jpg', nx = 9, ny = 6, debug = False, write_undistorted_output = False):
    if (debug):
        print("calc_distortion(camera_cal_img_path = {}, nx = {}, ny = {}, debug = {})".format(camera_cal_img_path, nx, ny, debug))

    import numpy as np
    import glob
    import cv2

    # Storing image path and image in a map together
    imagesMap = {}

    # Storing object points and image points
    obj_points = []
    img_points = []

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    # Looping over each image path in provided path for camera calculation images
    for img_path in glob.glob(camera_cal_img_path + '/*.' + camera_cal_img_file_type):
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

mtx, dist = calc_distortion(debug = True)
