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
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


theta = 0.05
rotating_matrix = np.array([[
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0],
    [0, 0, 1]
]])

Z_axis = np.array([[0.],[0.],[1.]])

def run():
    cam = cv2.VideoCapture('video/VIDEO0122.mp4')
    frame_idx = 0
    T = np.eye(4)
    T[:3,:3] = cameraMatrix
    match_stack_man_cur = match_stack_man
    tracking = False
    R_cur, t_cur = np.eye(3), np.zeros((3,1))
    while(cam.isOpened()):
        # Capturing each frame of our video stream
        ret, QueryImg = cam.read()
        frame_idx += 1
        if ret == True and frame_idx % 1 == 0:

            # grayscale image
            QueryImg = cv2.resize(QueryImg, (0,0), fx=0.2, fy=0.2)
            # cv2.imwrite("frames/%06d.jpg" % frame_idx, QueryImg)
            
            gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)
        
            # Detect Aruco markers
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
    
            # Outline all of the markers detected in our image
            # QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, borderColor=(0, 0, 255))
            # Require 15 markers before drawing axis
            if ids is not None and len(ids) > 3:
                # Display our image
                idx = np.where(ids == 98)
                if len(idx[0]) > 0::
                    idx = idx[0][0]
                    corners = np.array(corners).reshape((-1,2))
                    corners_sub = cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
                    corners_sub_undistort = cv2.undistortPoints(corners_sub, cameraMatrix, distCoeffs, P=cameraMatrix).reshape((-1,4,2))
                    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers([corners_sub.reshape((-1,4,2))[idx]], 1, cameraMatrix, distCoeffs) 
                    brvec, btvec = rvecs[0], tvecs[0]
                    if not tracking or frame_idx % 10 == 0:
                        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers([corners_sub.reshape((-1,4,2))[idx]], 1, cameraMatrix, distCoeffs) 
                        rvec, tvec = rvecs[0], tvecs[0]
                        R_cur, _ = cv2.Rodrigues(rvec)
                        t_cur = tvec.reshape((3,1))
                        tracking = True
                    else:
                        found_ids1 = []
                        found_corners1 = []
                        found_ids2 = []
                        found_corners2 = []
                        for i,id1 in enumerate(prev_ids):
                            idx = np.where(ids == id1[0])
                            if len(idx[0]) > 0:
                                idx = idx[0][0]
                                found_ids1.append(prev_ids[i])
                                found_corners1.append(prev_corners_sub[i])
                                found_ids2.append(ids[idx])
                                found_corners2.append(corners_sub_undistort[idx])
                        found_corners1 = np.array(found_corners1).reshape((-1,2))
                        found_corners2 = np.array(found_corners2).reshape((-1,2))
                        # print('----------------------')
                        # print(found_corners1)
                        # print(found_corners2)
                        # print('----------------------')
                        flag = False

                        if found_corners1.shape[0] >= 4:
                            H, _ = cv2.findHomography(found_corners1, found_corners2, method=cv2.RANSAC, ransacReprojThreshold=1)
                            # H, _ = cv2.findHomography(prev_corners_sub, corners_sub_undistort)
                            _, Rs, ts, ns = cv2.decomposeHomographyMat(H, cameraMatrix)
                            # print(Rs)
                            # print(ts)
                            # print('--------')
                            for R, t in zip(Rs, ts):
                                R_tmp = R @ R_cur
                                t_tmp = R @ t_cur + t.reshape((3,1))
                                z = cameraMatrix @ (R_tmp @ Z_axis + t_tmp)
                                z2d = z[:2] / z[2]
                                print(R_tmp, t_tmp)
                                if np.sum(z > 0) == 3 and 0 <= z2d[0] < gray.shape[1] and 0 <= z2d[1] < gray.shape[0] and t_tmp[2,0]:
                                    # print(R, t)
                                    # print(R_tmp, t_tmp)
                                    R_cur = R_tmp
                                    t_cur = t_tmp
                                    rvec, _ = cv2.Rodrigues(R_cur)
                                    tvec = t_cur.reshape(-1)
                                    tracking = True
                                    flag = True
                                    break
                        # print(R_cur, t_cur)
                        bR_cur, _ = cv2.Rodrigues(brvec)
                        bt_cur = btvec.reshape((3,1))
                        print(bR_cur, bt_cur)
                        print("--------------")
                        if not flag:
                            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers([corners_sub], 1, cameraMatrix, distCoeffs) 
                            rvec, tvec = rvecs[0], tvecs[0]
                            R_cur, _ = cv2.Rodrigues(rvec)
                            t_cur = tvec.reshape((3,1))
                            tracking = True

                    # print(frame_idx, tracking)

                    # QueryImg = aruco.drawAxis(QueryImg, cameraMatrix, distCoeffs, rvec, tvec, 1)
                    # axis_cur = (rotating_matrix @ axis_cur.T).T
                    imgpts, jac = cv2.projectPoints(axis, rvec, tvec, cameraMatrix, distCoeffs)
                    draw(QueryImg, corners, imgpts)
                    # match_stack_man_cur = (rotating_matrix @ match_stack_man_cur.T).T
                    # imgpts, jac = cv2.projectPoints(match_stack_man_cur, rvec, tvec, cameraMatrix, distCoeffs)
                    # drawMatchStickMan(QueryImg, imgpts)
                    prev_corners_sub = corners_sub_undistort
                    prev_ids = ids
            

            cv2.imshow('QueryImage', QueryImg)
    
        # Exit at the end of the video on the 'q' keypress
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()