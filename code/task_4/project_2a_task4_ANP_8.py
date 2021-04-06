#!/usr/bin/env python3

#####################################
# CSE 598: Perception in Robotics   #
# Project 2a by Aniket N Prabhu     #
# Task 4: Dense depth triangulation #
# Use Linux                         #
#####################################

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from os import chdir

''' Loading images and camera parameters '''
# Saving path to avoid calling it repeatedly:
currwd = __file__  # Current working directory
chng_str = "code/task_4/project_2a_task4_ANP_8.py"
image_path = currwd.replace(chng_str, "images/task_3_and_4/")

# Loading images
limg = cv2.imread(image_path + "left_8.png")
w_l, h_l = limg.shape[:2][::-1]
rimg = cv2.imread(image_path + "right_8.png")
w_r, h_r = rimg.shape[:2][::-1]

# Path to parameter files
cam_param_path = currwd.replace(chng_str, "parameters/")

# Getting the camera intrinsic parameters
def camparamcalc(parampath, param_file_name):
    cam_params = cv2.FileStorage(parampath + param_file_name, cv2.FILE_STORAGE_READ)
    K = cam_params.getNode("Intrinsic_Matrix").mat()
    D = cam_params.getNode("Distortion_Coefficients").mat()
    return K, D

# Left camera parameters
KL, DL = camparamcalc(cam_param_path, "left_camera_intrinsics.xml")

# Right camera parameters
KR, DR = camparamcalc(cam_param_path, "right_camera_intrinsics.xml")

# Rotation and translation matrices
stereo_param = cv2.FileStorage(cam_param_path + "stereo_calibration.xml", cv2.FILE_STORAGE_READ)
R = stereo_param.getNode("Rotation_matrix").mat()
T = stereo_param.getNode("Translation_vector").mat()

# Rectification and projection matrices
stereo_param = cv2.FileStorage(cam_param_path + "stereo_rectification.xml", cv2.FILE_STORAGE_READ)
rl = stereo_param.getNode("Left_camera_rotation_matrix").mat()
rr = stereo_param.getNode("Right_camera_rotation_matrix").mat()
pl = stereo_param.getNode("Left_camera_projection_matrix_rectified").mat()
pr = stereo_param.getNode("Right_camera_projection_matrix_rectified").mat()
q = stereo_param.getNode("Disparity-to-depth_mapping_matrix").mat()

''' Block match for each pixel on the images to obtain a disparity map '''

# Disparity map requires undistorted images
def imgRectified(img, K, D, rectmat, Knew, imgshape):
    mapx_rect, mapy_rect = cv2.initUndistortRectifyMap(K, D, rectmat, Knew, imgshape, cv2.CV_16SC2)  # rectified
    dist_rect = cv2.remap(img, mapx_rect, mapy_rect, cv2.INTER_LINEAR)
    return dist_rect

output_dir = currwd.replace("code/task_4/project_2a_task4_ANP_8.py", "output/task_4/")

dist_l_4_rect = imgRectified(limg, KL, DL, rl, pl, (w_l, h_l))
cv2.imshow('Rectified left_8.png', dist_l_4_rect)
cv2.imwrite(output_dir+"rectified_left_8.png", dist_l_4_rect)
dist_r_4_rect = imgRectified(rimg, KR, DR, rr, pr, (w_r, h_r))
cv2.imshow('Rectified left_8.png', dist_r_4_rect)
cv2.imwrite(output_dir+"rectified_right_8.png", dist_r_4_rect)
distl4rect_grey, distr4rect_grey = cv2.cvtColor(dist_l_4_rect, cv2.COLOR_BGR2GRAY), \
                                   cv2.cvtColor(dist_r_4_rect, cv2.COLOR_BGR2GRAY)

# Initializing Stereo Block Matching (StereoBM) class (Left Matcher)
stereosgbm = cv2.StereoSGBM_create(minDisparity = 9, numDisparities = 64, blockSize = -1, preFilterCap = 63, \
                                 uniquenessRatio = 15, speckleWindowSize = 10, disp12MaxDiff = 20, P1 = 1176, \
                                 P2 = 4704, mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY)  # Gotta change blocksize

# Setting up matcher for computing right-view disparity map
matcher = cv2.ximgproc.createRightMatcher(stereosgbm)

# We will use disparity filter map based on Weighted Least Squares
# Initializing Disparity WLS filter

l, s = 70000, 1.2  # lambda and sigma

disp_fil = cv2.ximgproc.createDisparityWLSFilter(stereosgbm)
disp_fil.setLambda(l)
disp_fil.setSigmaColor(s)

# computing disparity maps
disparity_l = stereosgbm.compute(dist_l_4_rect, dist_r_4_rect)
disparity_r = matcher.compute(dist_r_4_rect, dist_l_4_rect)
disparity_l, disparity_r = np.int16(disparity_l), np.int16(disparity_r)

# Applying the Disparity WLS filter on left_5.png
filtered_disp_l = disp_fil.filter(disparity_l, dist_l_4_rect, None, disparity_r)

# Normalizing maps for OpenCV visualization
disparity_l = cv2.normalize(disparity_l, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, \
                            dtype = cv2.CV_8U)
filtered_disp_l = cv2.normalize(filtered_disp_l, None, alpha = 0, beta = 255, \
                                norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)


# Plotting maps
cv2.imshow('Disparity map without Disparity WLS filter', disparity_l)  # Without Disparity WLS filter
cv2.imwrite(output_dir+"wo_disp_wls_8.png", disparity_l)
# plt.show()
cv2.imshow('Disparity map with Disparity WLS filter', filtered_disp_l)  # With Disparity WLS filter
cv2.imwrite(output_dir+"w_disp_wls_8.png", filtered_disp_l)
cv2.waitKey(0)

''' Calculate depth for each pixel using the disparity map '''
# Disparity map computed with Stereo SGBM needs to be scaled to 'float'
# and divided by 16 before being passed on to reprojectImageTo3D()
filtered_disp_l = filtered_disp_l.astype(np.float32) / 16.0
fil_disp_l_3d = cv2.reprojectImageTo3D(filtered_disp_l, q)
cv2.imshow('Depth image', fil_disp_l_3d)
cv2.imwrite(output_dir+"depth_image_8.png", fil_disp_l_3d)
cv2.waitKey(0)