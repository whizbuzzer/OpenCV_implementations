#!/usr/bin/env python3

################################################
# CSE 598: Perception in Robotics              #
# Project 2a by Aniket N Prabhu                #
# Task 1: Pinhole camera model and calibration #
# Use Linux                                    #
################################################

import numpy as np
import cv2
from os import chdir

def calibrate_camera(left_or_right, images, path):
    # Lists for storing 2D (image) points:
    cornarray = []

    # To create 3d (object) points:
    objpoints = np.zeros((6 * 9, 3), np.float32)
    objpoints[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Lists for strong 3D points:
    objp_array = []

    # Finding object points and image points for left camera:
    for i in range(len(images)):
        img = cv2.imread(path+images[i])
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Gets rid of RGB data which is not required to find points
        h, w = img.shape[:2]  # 480x640
        retval, corners = cv2.findChessboardCorners(grey, (9,6), None)
        if retval == 1:
            objp_array.append(objpoints)
            cornarray.append(corners)

    '''Calculating camera intrinsic parameters for left camera: '''
    ter, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objp_array, cornarray, grey.shape[::-1], None, None)

    if left_or_right == "l":
        xml_file = "left_camera_intrinsics.xml"
    elif left_or_right == "r":
        xml_file = "right_camera_intrinsics.xml"

    cam_params = cv2.FileStorage(currwd.replace("code/task_1/project_2a_task1_ANP.py", "parameters/" + xml_file), cv2.FileStorage_WRITE)
    cam_params.write("Intrinsic_Matrix", mtx)
    cam_params.write("Distortion_Coefficients", dist)

    '''Checking calibration results for left camera: '''
    newcammat, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcammat, (w, h), 5)
    img = cv2.imread(path + images[2])  # left_2.png
    dist = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    x, y, w, h = roi
    dist = dist[y:y + h, x:x + w]
    return dist


if __name__ == "__main__":
    ''' Loading images and extracting 3D-to-2D point correspondences '''
    # Saving path to avoid calling it repeatedly:
    currwd = __file__  # Current working directory
    path = currwd.replace("code/task_1/project_2a_task1_ANP.py", "images/task_1/")

    # path = r'/home/whizbuzzer/CSE 598/Assignments/Project 2a/project_2a/images/task_1/'

    # Storing image names in a list to call them later in the for loop:
    images_l, images_r = [], []  # Lists for storing left and right images
    for i in range(11):
        images_l.append(f"left_{i}.png")
        images_r.append(f"right_{i}.png")


    dist_l = calibrate_camera("l", images_l, path)
    dist_r = calibrate_camera("r", images_r, path)

    chdir(currwd.replace("code/task_1/project_2a_task1_ANP.py", "output/task_1/"))
    cv2.imshow("calibration for left image 2", dist_l)
    cv2.imwrite("left_2_calibrated.png", dist_l)
    cv2.waitKey(0)
    cv2.imshow("calibration for right image 2", dist_r)
    cv2.imwrite("right_2_calibrated.png", dist_r)
    cv2.waitKey(0)
