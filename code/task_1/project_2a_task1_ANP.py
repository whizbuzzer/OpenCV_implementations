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

''' Loading images and extracting 3D-to-2D point correspondences '''
# Saving path to avoid calling it repeatedly:
currwd = __file__  # Current working directory
path = currwd.replace("code/task_1/project_2a_task1_ANP.py", "images/task_1/")

# path = r'/home/whizbuzzer/CSE 598/Assignments/Project 2a/project_2a/images/task_1/'

# Storing image names in a list to call them later in the for loop:
images_l = ["left_0.png", "left_1.png", "left_2.png", "left_3.png", "left_4.png",
            "left_5.png", "left_6.png", "left_7.png", "left_8.png", "left_9.png", "left_10.png"]        # left
images_r = ["right_0.png", "right_1.png", "right_2.png", "right_3.png", "right_4.png",
            "right_5.png", "right_6.png", "right_7.png", "right_8.png", "right_9.png", "right_10.png"]  # right

# Only one n_images need since both left and right have equal n(image)
n_images = len(images_l) 

# Lists for storing 2D (image) points:
cornarray_l = []  
cornarray_r = []

# To create 3d (object) points:
objpoints = np.zeros((6*9, 3), np.float32)
objpoints[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Lists for strong 3D points:
objp_array_l = []
objp_array_r = []

# Finding object points and image points for left camera:
for i in range(n_images):
    limg = cv2.imread(path+images_l[i])
    grey_l = cv2.cvtColor(limg, cv2.COLOR_BGR2GRAY)
    h_l, w_l = limg.shape[:2]  # 480x640
    retval_l, corners_l = cv2.findChessboardCorners(grey_l, (9,6), None)
    if retval_l == 1:
        objp_array_l.append(objpoints)
        cornarray_l.append(corners_l)

'''Calculating camera intrinsic parameters for left camera: '''
ter_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objp_array_l, cornarray_l, grey_l.shape[::-1], None, None)

l_cam_params = cv2.FileStorage(currwd.replace("code/task_1/project_2a_task1_ANP.py", "parameters/left_camera_intrinsics.xml"), cv2.FileStorage_WRITE)
l_cam_params.write("Intrinsic_Matrix", mtx_l)
l_cam_params.write("Distortion_Coefficients", dist_l)

'''Checking calibration results for left camera: '''
newcammat_l, roi_l = cv2.getOptimalNewCameraMatrix(mtx_l, dist_l, (w_l, h_l), 0, (w_l, h_l))
mapx_l, mapy_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, None, newcammat_l, (w_l, h_l), 5)
limg = cv2.imread(path+images_l[2])  # left_2.png
dist_l = cv2.remap(limg, mapx_l, mapy_l, cv2.INTER_LINEAR)
x_l, y_l, w_l, h_l = roi_l
dist_l = dist_l[y_l:y_l+h_l, x_l:x_l+w_l]

# Finding object points and image points for right camera:
for i in range(n_images):
    rimg = cv2.imread(path+images_r[i])
    grey_r = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)
    h_r, w_r = rimg.shape[:2]
    retval_r, corners_r = cv2.findChessboardCorners(grey_r, (9,6), None)
    if retval_r == 1:
        objp_array_r.append(objpoints)
        cornarray_r.append(corners_r)

'''Calculating camera intrinsic parameters for right camera: '''
ter_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objp_array_r, cornarray_r, (grey_r.shape[::-1]), None, None)

r_cam_params = cv2.FileStorage(currwd.replace("code/task_1/project_2a_task1_ANP.py", "parameters/right_camera_intrinsics.xml"), cv2.FileStorage_WRITE)
r_cam_params.write("Intrinsic_Matrix", mtx_r)
r_cam_params.write("Distortion_Coefficients", dist_r)

'''Checking calibration results for left camera: '''
newcammat_r, roi_r = cv2.getOptimalNewCameraMatrix(mtx_r, dist_r, (w_r, h_r), 0, (w_r, h_r))
mapx_r, mapy_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, None, newcammat_r, (w_r, h_r), 5)
rimg = cv2.imread(path+images_r[2])  # right_2.png
dist_r = cv2.remap(rimg, mapx_r, mapy_r, cv2.INTER_LINEAR)
x_r, y_r, w_r, h_r = roi_r
dist_r = dist_r[y_r:y_r+h_r, x_r:x_r+w_r]

chdir(currwd.replace("code/task_1/project_2a_task1_ANP.py", "output/task_1/"))
cv2.imshow("calibration for left image 2", dist_l)
cv2.imwrite("left_2_calibrated.png", dist_l)
cv2.waitKey(0)
cv2.imshow("calibration for right image 2", dist_r)
cv2.imwrite("right_2_calibrated.png", dist_r)
cv2.waitKey(0)
