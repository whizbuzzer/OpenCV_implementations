#!/usr/bin/env python3

######################################
# CSE 598: Perception in Robotics    #
# Project 2a by Aniket N Prabhu      #
# Task 3: Sparse depth triangulation #
# Use Linux                          #
######################################

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from os import chdir

''' Loading images and intrinsic parameters: '''
# Saving path to avoid calling it repeatedly:
currwd = __file__  # Current working directory
image_path = currwd.replace("code/task_3/project_2a_task3_ANP.py", "images/task_3_and_4/")

# Storing image names in a list to call them later in the for loop:
images_l = ["left_0.png", "left_1.png", "left_2.png", "left_3.png", "left_4.png",
            "left_5.png", "left_6.png", "left_7.png", "left_8.png", "left_9.png", "left_10.png"]        # left
images_r = ["right_0.png", "right_1.png", "right_2.png", "right_3.png", "right_4.png",
            "right_5.png", "right_6.png", "right_7.png", "right_8.png", "right_9.png", "right_10.png"]  # right

# Storing grayscale images:
grey_l = []
grey_r = []

# Only one n_images need since both left and right have equal n(image)
n_images = len(images_l) 

'''Calculating 3D-to-2D point correspondences and image dimensions: '''
# Function to calculate 3D point array, 2D point array, height and width of an image:
def grey(imgpath, imgarr):
    grey_img = []
    for i in range(len(imgarr)):
        img = cv2.imread(imgpath+imgarr[i])
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grey_img.append(grey)
        h, w = img.shape[:2]
    return grey_img, h, w

grey_l, h_l, w_l = grey(image_path, images_l)
grey_r, h_r, w_r = grey(image_path, images_r)

# Getting the camera intrinsic parameters:
cam_param_path = currwd.replace("code/task_3/project_2a_task3_ANP.py", "parameters/")

def camparamcalc(parampath, param_file_name):
    cam_params = cv2.FileStorage(parampath + param_file_name, cv2.FILE_STORAGE_READ)
    K = cam_params.getNode("Intrinsic_Matrix").mat()
    D = cam_params.getNode("Distortion_Coefficients").mat()
    return K, D

# Left camera parameters:
KL, DL = camparamcalc(cam_param_path, "left_camera_intrinsics.xml")

# Right camera parameters:
KR, DR = camparamcalc(cam_param_path, "right_camera_intrinsics.xml")

# Rotation and translation matrices:
stereo_param = cv2.FileStorage(cam_param_path + "stereo_calibration.xml", cv2.FILE_STORAGE_READ)
R = stereo_param.getNode("Rotation_matrix").mat()
T = stereo_param.getNode("Translation_vector").mat()

''' Detect features '''
# Getting undistorted points:
mapx_l, mapy_l = cv2.initUndistortRectifyMap(KL, DL, None, KL, (w_l, h_l), 5, cv2.CV_16SC2)  # left_0.png
dist_l = cv2.remap(grey_l[0], mapx_l, mapy_l, cv2.INTER_LINEAR)

mapx_r, mapy_r = cv2.initUndistortRectifyMap(KR, DR, None, KR, (w_r, h_r), 5, cv2.CV_16SC2)  # right_0.png
dist_r = cv2.remap(grey_r[0], mapx_r, mapy_r, cv2.INTER_LINEAR)

# Initializing ORB class
orb = cv2.ORB_create()

# Detecting keypoints
keypoints_l = orb.detect(dist_l, None)
# print(keypoints_l)
keypoints_r = orb.detect(dist_r, None)

# Selecting local maxima among keypoints
def nms(keypoints, scan_rad):
    locmax, corrdesc = [], []
    n = len(keypoints)
    for i in range(n):
        kpi = keypoints[i].pt
        max = 1  # flag
        for j in range(n):
            if i == j:
                continue
            kpj = keypoints[j].pt
            kp_rad = np.math.sqrt((kpj[0] - kpi[0])**2 + (kpj[1] - kpi[1])**2)
            if kp_rad < scan_rad and keypoints[i].response < keypoints[j].response:
                max = 0
                break

        if max == 1:
            locmax.append(keypoints[i])
    return locmax  #, np.array(corrdesc)

keypoints_l = nms(keypoints_l, 7)
# print(type(desc_l), type(desc_l[0]))
keypoints_r = nms(keypoints_r, 7)
# print(len(keypoints_l), len(keypoints_r))

# To ensure that both keypoint lists are of the same length
kp_len = min(len(keypoints_l), len(keypoints_r))
keypoints_l = keypoints_l[:kp_len]
keypoints_r = keypoints_r[:kp_len]
# print(kp_len)


# Computing descriptors
keypoints_l, desc_l = orb.compute(dist_l, keypoints_l)
keypoints_r, desc_r = orb.compute(dist_r, keypoints_r)

# Drawing keypoints locations
output_dir = currwd.replace("code/task_3/project_2a_task3_ANP.py", "output/task_3/")
img_l = cv2.drawKeypoints(dist_l, keypoints_l, None, color=(255, 0, 0), flags = 0)
plt.imshow(img_l)
cv2.imwrite(output_dir+"left_0_features.png", img_l)
plt.show()
img_r = cv2.drawKeypoints(dist_r, keypoints_r, None, color=(0, 0, 255), flags = 0)
cv2.imwrite(output_dir+"right_0_features.png", img_r)
plt.imshow(img_r)
plt.show()

''' Matching features '''
"""Given two sets of descriptors, Brute-Force Matcher (BFMatcher) class scans one set
   and for each descriptor in that set, it finds the closest matching descriptor in the second set
"""
# Initializing BFMatcher class
bfm = cv2.BFMatcher(normType = cv2.NORM_HAMMING, crossCheck = True)

# Matching left and right keypoint descriptors
desc_match = bfm.match(desc_l, desc_r)

# Getting good matches by eliminating the false/bad matches
desc_match_sorted = sorted(desc_match, key = lambda x: x.distance)
good_desc_match = []
for match in desc_match_sorted:
    img1_idx = match.queryIdx  # left_0.png descriptor indices
    img2_idx = match.trainIdx  # right_0.png descriptor indices
    y_coord_l = keypoints_l[img1_idx].pt[1]
    y_coord_r = keypoints_r[img2_idx].pt[1]
    if abs(y_coord_r - y_coord_l) < 10:
        good_desc_match.append(match)

# Drawing the matches
desc_match_drawn = cv2.drawMatches(img_l, keypoints_l, img_r, keypoints_r,
                                   good_desc_match, None)
plt.imshow(desc_match_drawn)
cv2.imwrite(output_dir+"matches_drawn.png", desc_match_drawn)
plt.show()

# Projection matrices:
pl = np.concatenate((np.identity(3), np.zeros((3,1))), axis=1)  # [I|0]
pr = np.concatenate((R, T), axis=1)                             # [R|T]

# Turning keypoints into pixel coordinates
pix_kp_l, pix_kp_r = [], []
for match in good_desc_match:
    img1_idx = match.queryIdx  # left_0.png descriptor indices
    img2_idx = match.trainIdx  # right_0.png descriptor indices
    pix_kp_l.append(keypoints_l[img1_idx].pt)
    pix_kp_r.append(keypoints_r[img2_idx].pt)

# Normalizing coordinates for triangulation
norm_kp_l = cv2.undistortPoints(np.array(pix_kp_l), KL, DL)
norm_kp_r = cv2.undistortPoints(np.array(pix_kp_r), KR, DR)
triang_pts = cv2.triangulatePoints(pl, pr, norm_kp_l, norm_kp_r)

''' Checking sparse depth results '''
# Conversion to 3D points
points_3d = cv2.convertPointsFromHomogeneous(triang_pts.T).reshape(-1, 3)
x, y, z = [], [], []
for i in range(len(points_3d)):
    p = points_3d[i]
    x.append(p[0])
    y.append(p[1])
    z.append(p[2])

# Plotting the 3D points
plt.figure()
ax = plt.gca(projection = '3d')
ax.scatter3D(x, y, z)
plt.show()