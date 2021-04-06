#!/usr/bin/env python3

###############################################
# CSE 598: Perception in Robotics             #
# Project 2a by Aniket N Prabhu               #
# Task 2: Stereo calibration and rectfication #
# Use Linux                                   #
###############################################

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from os import chdir

''' Loading images and intrinsic parameters: '''
# Saving path to avoid calling it repeatedly:
currwd = __file__  # Current working directory
image_path = currwd.replace("code/task_2/project_2a_task2_ANP.py", "images/task_2/")
# print(path)

# Storing image names in a list to call them later in the for loop:
images_l = ["left_0.png", "left_1.png"]                     # left
images_r = ["right_0.png", "right_1.png"]                   # right

# Only one n_images need since both left and right have equal n(image)
n_images = len(images_l)

# Lists for storing 2D (image) points:
cornarray_l, cornarray_r = [], []

# Lists for strong 3D points:
objp_array_l, objp_array_r = [], []

# To create 3d (object) points:
objpoints = np.zeros((6*9, 3), np.float32)
objpoints[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

'''Calculating 3D-to-2D point correspondences and image dimensions: '''
# Function to calculate 3D point array, 2D point array, height and width of an image:
def objimgpts(imgpath, imgarr, objarr):
    objp_array, cornarray = [], []
    for i in range(len(imgarr)):
        img = cv2.imread(imgpath+imgarr[i])
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        retval, corners = cv2.findChessboardCorners(grey, (9,6), None)  # Calibration board is 9x6
        if retval == 1:
            objp_array.append(objarr)
            cornarray.append(corners)
    return objp_array, cornarray, h, w

objp_array_l, cornarray_l, h_l, w_l = objimgpts(image_path, images_l, objpoints)
objp_array_r, cornarray_r, h_r, w_r = objimgpts(image_path, images_r, objpoints)
if objp_array_l == objp_array_r:
    objp_array = objp_array_l

limg = cv2.imread(image_path + "left_0.png")
rimg = cv2.imread(image_path + "right_0.png")

# Getting the camera intrinsic parameters:
intrin_param_path = currwd.replace("code/task_2/project_2a_task2_ANP.py", "parameters/")

def camparamcalc(parampath, param_file_name):
    cam_params = cv2.FileStorage(parampath + param_file_name, cv2.FILE_STORAGE_READ)
    K = cam_params.getNode("Intrinsic_Matrix").mat()
    D = cam_params.getNode("Distortion_Coefficients").mat()
    return K, D

# Left camera parameters:
KL, DL = camparamcalc(intrin_param_path, "left_camera_intrinsics.xml")

# Right camera parameters:
KR, DR = camparamcalc(intrin_param_path, "right_camera_intrinsics.xml")

''' Calibrating the stereo camera '''
rms_err, cammat_l, distcoeff_l, cammat_r, distcoeff_r, R, T, E, F = cv2.stereoCalibrate(
    [objp_array[0]], [cornarray_l[0]], [cornarray_r[0]], KL, DL, KR, DR, (h_l, w_l), cv2.CALIB_FIX_INTRINSIC)
# print(retval)
'''
cammat_l = left camera intrinsic matrix
cammat_r = right camera intrinsic matrix
distcoeff_l = left camera distant coefficients
distcoeff_r = right camera distant coefficients
cv2.CALIB_FIX_INTRINSIC flag fixes camera intrinsic matrix and distance coefficients
so that only R, T, E and F matrices are estimated
'''

stereo_calib_params = cv2.FileStorage(currwd.replace("code/task_2/project_2a_task2_ANP.py", "parameters/stereo_calibration.xml"),
    cv2.FileStorage_WRITE)
stereo_calib_params.write("RMS_error", rms_err)
stereo_calib_params.write("Rotation_matrix", R)
stereo_calib_params.write("Translation_vector", T)
stereo_calib_params.write("Essential_matrix", E)
stereo_calib_params.write("Fundamental_matrix", F)

# Normalizing the translation vector by multiplying with given baseline width (62mm):
T *= 0.62

''' Checking calibration results (pre-rectification)'''
# Triangulation requires undistorted points:
und_limg = cv2.undistortPoints(cornarray_l[0], KL, DL)
und_rimg = cv2.undistortPoints(cornarray_r[0] , KR, DR)

# Triangulation also requires projection matrices:
P1 = np.hstack((np.identity(3, dtype=float), np.zeros((3,1), dtype=float)))
# print(P1)
P2 = np.hstack((R, T))
# print(P2)
triang_pts = cv2.triangulatePoints(P1, P2, und_limg, und_rimg)
# print(triang_pts.shape)


''' Rectifying the stereo camera '''
rl, rr, pl, pr, q, roil, roir = cv2.stereoRectify(KL, DL, KR, DR, (h_l, w_l), R, T)
stereo_rect_params = cv2.FileStorage(currwd.replace("code/task_2/project_2a_task2_ANP.py", "parameters/stereo_rectification.xml"), 
    cv2.FileStorage_WRITE)
stereo_rect_params.write("Left_camera_rotation_matrix", rl)   # Also called rectification matrix
stereo_rect_params.write("Right_camera_rotation_matrix", rr)  # Also called rectification matrix
stereo_rect_params.write("Left_camera_projection_matrix_rectified", pl)
stereo_rect_params.write("Right_camera_projection_matrix_rectified", pr)
stereo_rect_params.write("Disparity-to-depth_mapping_matrix", q)
stereo_rect_params.write("Left_camera_region-of-interest", roil)
stereo_rect_params.write("Right_camera_region-of-interest", roir)


''' Checking the rectification results '''
def getimg(img, K, D, rectmat, Knew, imgshape):
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, K, imgshape, cv2.CV_16SC2)   # check with cammat as well as newcammat
    mapx_rect, mapy_rect = cv2.initUndistortRectifyMap(K, D, rectmat, Knew, imgshape, cv2.CV_16SC2)  # rectified
    dist = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    dist_rect = cv2.remap(img, mapx_rect, mapy_rect, cv2.INTER_LINEAR)
    return dist, dist_rect

# rimg = cv2.imread(image_path + "left_1.png")

dist_l_0, dist_l_0_rect = getimg(limg, KL, DL, rl, pl, (w_l, h_l))
dist_r_0, dist_r_0_rect = getimg(rimg, KR, DR, rr, pr, (w_r, h_r))

output_dir = currwd.replace("code/task_2/project_2a_task2_ANP.py", "output/task_2/")
chdir(output_dir)
cv2.imshow("calibration for left image 0", dist_l_0)
cv2.imwrite("left_0_calibrated.png", dist_l_0)
cv2.waitKey(0)
cv2.imshow("calibration for left image 0 rectified", dist_l_0_rect)
cv2.imwrite("left_0_calibrated_rectified.png", dist_l_0_rect)
cv2.waitKey(0)
cv2.imshow("calibration for right image 0", dist_r_0)
cv2.imwrite("right_0_calibrated.png", dist_r_0)
cv2.waitKey(0)
cv2.imshow("calibration for right image 1 rectified", dist_r_0_rect)
cv2.imwrite("right_0_calibrated_rectified.png", dist_r_0_rect)
cv2.waitKey(0)

''' 3D plotting camera position '''

# For individual camera polygon:
def campolygon(ax, R, T=np.zeros((1, 3))):

    # Coordinates for polygon vertices:
    x = [1, -1, -1, 1, 0]
    y = [1, 1, -1, -1, 0]
    z = [4, 4, 4, 4, 0]
    vertlist = list(zip(x, y, z))
    vertarray = np.array(vertlist, np.float32)

    # Polygon coordinates for camera:
    vertarray *= 0.75  # For scaling
    vertarray = np.asarray(np.asmatrix(R) * vertarray.T + T.T).T
    verts = [[vertarray[0], vertarray[1], vertarray[2], vertarray[3]], 
             [vertarray[0], vertarray[1], vertarray[4]], [vertarray[1], vertarray[2], vertarray[4]],
             [vertarray[2], vertarray[3], vertarray[4]], [vertarray[3], vertarray[0], vertarray[4]]]
    
    # 3D-plotting then camera polygon:
    ax.add_collection3d(Poly3DCollection(verts, linewidths = 1, edgecolors = 'k', alpha = 0.0))
    

# For plotting camera's line of vision
def normcalc(Rx):
    norm1c = np.ravel(np.asmatrix(Rx) * np.array([[0], [0], [6]]))
    # print(np.array([[0], [0], [6]]).shape)
    norm1v = np.linspace(0, norm1c, 100)
    return norm1c, norm1v

# Making the complete plots
def camplot(R1, R2, T, triang_pts):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    norm1c, norm1v = normcalc(R1)
    ax.plot3D(norm1v[:, 0], norm1v[:, 1], norm1v[:, 2])

    campolygon(ax, R1)

    norm1c, norm1v = normcalc(R2)
    norm1v += T.reshape(1, 3)
    ax.plot3D(norm1v[:, 0], norm1v[:, 1], norm1v[:, 2])

    campolygon(ax, R2, T.reshape(1,3))

    # Conversion to non-homogeneous points:
    xnh = triang_pts[0]/triang_pts[3]
    ynh = triang_pts[1]/triang_pts[3]
    znh = triang_pts[2]/triang_pts[3]

    ax.scatter(xnh, ynh, znh, c=znh)
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-10, 10])

    return fig

# Unrectified plot:
camplot(np.identity(3), R, T, triang_pts)
plt.savefig(output_dir+"task_2_3d_plot.png")
plt.show()

# Rectified plot:
''' Checking calibration results (post-rectification) '''
# Triangulation requires undistorted points:
und_limg = cv2.undistortPoints(cornarray_l[0], KL, DL, pl, rl)
und_rimg = cv2.undistortPoints(cornarray_r[0] , KR, DR, pr, rr)

# Triangulation also requires projection matrices:
P1 = np.hstack((np.identity(3, dtype=float), np.zeros((3,1), dtype=float)))
# print(P1)
P2 = np.hstack((R, T))
# print(P2)
triang_pts = cv2.triangulatePoints(P1, P2, und_limg, und_rimg)
# print(triang_pts.shape)
# triang_pts = cv2.triangulatePoints(pl, pr, und_limg, und_rimg)
camplot(rl, rr, T, triang_pts)
plt.savefig(output_dir+"task_2_3d_plot_rectified.png")
plt.show()
