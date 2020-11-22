# The following code is used to watch a video stream, detect Aruco markers, and use
# a set of markers to determine the posture of the camera in relation to the plane
# of markers.
#
# Assumes that all markers are on the same plane, for example on the same piece of paper
#
# Requires camera calibration (see the rest of the project for example calibration)

import numpy as np
import cv2
import cv2.aruco as aruco
import os
import pickle

# # Check for camera calibration data
# if not os.path.exists('./calibration.pckl'):
#     print("You need to calibrate the camera you'll be using. See calibration project directory for details.")
#     exit()
# else:
#     f = open('calibration.pckl', 'rb')
#     (cameraMatrix, distCoeffs, _, _) = pickle.load(f)
#     f.close()
#     if cameraMatrix is None or distCoeffs is None:
#         print("Calibration issue. Remove ./calibration.pckl and recalibrate your camera with CalibrateCamera.py.")
#         exit()


# cameraMatrix = np.array(
#     [[650.04859347,   0,         279.39064767],
#     [  0,         649.79925731, 354.21513032],
#     [  0,           0,           1.        ]])
# distCoeffs = np.array([[ 0.29659968, -1.64415153, -0.03426771, -0.01672571,  3.98012381]])

cameraMatrix = np.array([
    [643.86657406,   0.,         302.        ],
    [  0.,        639.90581685, 402.5       ],
    [  0.,           0.,           1.        ]])
distCoeffs = np.array([[0.,0.,0.,0.,0.]])


# Constant parameters used in Aruco methods
ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_6X6_1000)

# # Create grid board object we're using in our stream
# board = aruco.GridBoard_create(
#         markersX=2,
#         markersY=2,
#         markerLength=0.09,
#         markerSeparation=0.01,
#         dictionary=ARUCO_DICT)

# Create vectors we'll be using for rotations and translations for postures
rvecs, tvecs = None, None

cam = cv2.VideoCapture('video/VIDEO0122.mp4')
T = np.eye(4)

axis = np.float32([[-0.5,-0.5,0], [-0.5,0.5,0], [0.5,0.5,0], [0.5,-0.5,0],
                   [-0.5,-0.5,1],[-0.5,0.5,1],[0.5,0.5,1],[0.5,-0.5,1] ])

def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-1)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),2)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),1)
    return img

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('aruco.mp4', fourcc, 30.0, (384,216))
while(cam.isOpened()):
    # Capturing each frame of our video stream
    ret, QueryImg = cam.read()
    if ret == True:
        # grayscale image
        QueryImg = cv2.resize(QueryImg, (0,0), fx=0.2, fy=0.2)
        gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)
    
        # Detect Aruco markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
  

        # Refine detected markers
        # Eliminates markers not part of our board, adds missing markers to the board
        # corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(
        #         image = gray,
        #         board = board,
        #         detectedCorners = corners,
        #         detectedIds = ids,
        #         rejectedCorners = rejectedImgPoints,
        #         cameraMatrix = cameraMatrix,
        #         distCoeffs = distCoeffs)   

        ###########################################################################
        # TODO: Add validation here to reject IDs/corners not part of a gridboard #
        ###########################################################################

        # Outline all of the markers detected in our image
        QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, borderColor=(0, 0, 255))
        # Require 15 markers before drawing axis
        if ids is not None and len(ids) > 3:
        #     # Estimate the posture of the gridboard, which is a construction of 3D space based on the 2D video 
        #     #pose, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, cameraMatrix, distCoeffs)
        #     #if pose:
        #     #    # Draw the camera posture calculated from the gridboard
        #     #    QueryImg = aruco.drawAxis(QueryImg, cameraMatrix, distCoeffs, rvec, tvec, 0.3)
        #     # Estimate the posture per each Aruco marker
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 1, cameraMatrix, distCoeffs)           
        #     for rvec, tvec in zip(rvecs, tvecs):
        #             QueryImg = aruco.drawAxis(QueryImg, cameraMatrix, distCoeffs, rvec, tvec, 1)
        # Display our image

        idx = np.where(ids == 98)
        if idx[0]:
            idx = idx[0]
            rvec, tvec = rvecs[idx], tvecs[idx]
            R, _ = cv2.Rodrigues(rvec)
            
            # T_cur = np.eye(4)
            # T_cur[:3,:3] = R
            # T_cur[:3,3] = tvec
            # T_cur_inv = np.linalg.inv(T_cur)
            QueryImg = aruco.drawAxis(QueryImg, cameraMatrix, distCoeffs, rvec, tvec, 1)
            imgpts, jac = cv2.projectPoints(axis, rvec, tvec, cameraMatrix, distCoeffs)
            draw(QueryImg, corners, imgpts)

        cv2.imshow('QueryImage', QueryImg)
        out.write(QueryImg.astype(np.uint8))
 
    # Exit at the end of the video on the 'q' keypress
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()
out.release()