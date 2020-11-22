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


axis = np.float32([[-0.5,-0.5,0], [-0.5,0.5,0], [0.5,0.5,0], [0.5,-0.5,0],
                   [-0.5,-0.5,1],[-0.5,0.5,1],[0.5,0.5,1],[0.5,-0.5,1] ])

match_stack_man = np.float32([
    [-1,0,2], [1,0,2], [1,0,3], [1,0,4], [-1,0,4], [-1,0,3],
    [0,0,3], [0,0,1], [-1,0,0], [1,0,0]
]) * 0.6

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


def drawMatchStickMan(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    points = [tuple(imgpt) for imgpt in imgpts]
    img = cv2.line(img, points[0], points[1], (255,0,0), 1)
    img = cv2.line(img, points[6], points[7], (255,0,0), 1)
    img = cv2.line(img, points[7], points[8], (255,0,0), 1)
    img = cv2.line(img, points[7], points[9], (255,0,0), 1)
    img = cv2.line(img, points[2], points[3], (255,0,0), 1)
    img = cv2.line(img, points[3], points[4], (255,0,0), 1)
    img = cv2.line(img, points[4], points[5], (255,0,0), 1)
    img = cv2.line(img, points[5], points[2], (255,0,0), 1)
    return img

cameraMatrix = np.array(
    [[650.04859347,   0,         279.39064767],
    [  0,         649.79925731, 354.21513032],
    [  0,           0,           1.        ]])
distCoeffs = np.array([[ 0.29659968, -1.64415153, -0.03426771, -0.01672571,  3.98012381]])


# cameraMatrix = np.array([
#     [643.86657406,   0.,         302.        ],
#     [  0.,        639.90581685, 402.5       ],
#     [  0.,           0.,           1.        ]])
# distCoeffs = np.array([[0.,0.,0.,0.,0.]])

# Constant parameters used in Aruco methods
ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_6X6_1000)

theta = 0.05
rotating_matrix = np.array([[
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0],
    [0, 0, 1]
]])

def drawStandingMark(img, corners, rvec, tvec, T, frame_idx):
    h, w = img.shape[:2]
    pts = np.array([[-0.5,0,1], [0.5,0,1], [0.5,0,0], [-0.5,0,0]])
    pts = (T @ pts.T).T

    time1_ = 30
    t1 = frame_idx % time1_
    v = 0.1
    g = 2 * v / time1_
    dz = v * t1 - 0.5 * g * t1 * t1

    r = 2
    time2_ = 150
    t2 = frame_idx % time2_
    omega = 2 * np.pi / time2_
    dx = -np.cos(r * omega * t2)
    dy = np.sin(r * omega * t2)

    # pdb.set_trace()
    pts[:,0] = pts[:,0] + dx
    pts[:,1] = pts[:,1] + dy
    pts[:,2] = pts[:,2] + dz

    imgpts, _ = cv2.projectPoints(pts,
                              rvec, tvec, cameraMatrix, distCoeffs)
    H, _ = cv2.findHomography(corners, imgpts)
    warped = cv2.warpPerspective(img, H, (w,h))
    mask = np.zeros((h,w), np.uint8)
    pts = imgpts.reshape((4,2)).astype(np.uint)
    cv2.fillPoly(mask, pts = [pts], color=(255,255,255))
    mask = (mask[...,np.newaxis]/255).astype(np.uint)
    rst = img * (1-mask) + warped * mask
    return rst


def run():
    cam = cv2.VideoCapture('video/VIDEO0122.mp4')
    frame_idx = 0
    T = np.eye(3)
    match_stack_man_cur = match_stack_man
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('ar.mp4', fourcc, 30.0, (384,216))
    while(cam.isOpened()):
        # Capturing each frame of our video stream
        ret, QueryImg = cam.read()
        if ret == True:
            # grayscale image
            QueryImg = cv2.resize(QueryImg, (0,0), fx=0.2, fy=0.2)
            # cv2.imwrite("frames/%06d.jpg" % frame_idx, QueryImg)
            frame_idx += 1
            gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)
        
            # Detect Aruco markers
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
    
            # Outline all of the markers detected in our image
            # QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, borderColor=(0, 0, 255))
            # Require 15 markers before drawing axis
            if ids is not None and len(ids) > 3:
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 1, cameraMatrix, distCoeffs)           
            
                # Display our image
                idx = np.where(ids == 98)
                if len(idx[0]) > 0:
                    idx = idx[0][0]
                    rvec, tvec = rvecs[idx], tvecs[idx]
                    
                    # match_stack_man_cur = (rotating_matrix @ match_stack_man_cur.T).T
                    # imgpts, jac = cv2.projectPoints(match_stack_man_cur, rvec, tvec, cameraMatrix, distCoeffs)
                    # drawMatchStickMan(QueryImg, imgpts)

                    QueryImg = drawStandingMark(QueryImg, corners[idx], rvec, tvec, T, frame_idx)
                    T = rotating_matrix @ T


            cv2.imshow('QueryImage', QueryImg.astype(np.uint8))
            out.write(QueryImg.astype(np.uint8))
    
        # Exit at the end of the video on the 'q' keypress
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cam.release()
    out.release()


if __name__ == "__main__":
    run()