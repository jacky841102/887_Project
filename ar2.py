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

def draw(img, corners, rvec, tvec, T, frame_idx, init_img, match_man_mask, init_rvec, init_tvec, HH, init_img2):
    h, w = img.shape[:2]
    pts = np.array([[-0.5,0,1], [0.5,0,1], [0.5,0,0], [-0.5,0,0]])
    pts[:,2] = pts[:,2] * 2
    
    pts = (T @ pts.T).T

    time1_ = 20
    t1 = frame_idx % time1_
    v = 0.1
    g = 2 * v / time1_
    dz = v * t1 - 0.5 * g * t1 * t1

    r = 2
    time2_ = 150
    t2 = frame_idx % time2_
    omega = 2 * np.pi / time2_
    dx = -np.cos(omega * t2) * r
    dy = np.sin(omega * t2) * r

    pts[:,0] = pts[:,0] - 2.5
    pts[:,0] = pts[:,0] + dx
    # pts[:,1] = pts[:,1] + dy
    pts[:,2] = pts[:,2] + dz

    R1, _ = cv2.Rodrigues(init_rvec)
    t1 = init_tvec.reshape((3,1))

    R2, _ = cv2.Rodrigues(rvec)
    t2 = tvec.reshape((3,1))
    R_1to2 = R2 @ R1.T;
    t_1to2 = R2 @ (-R1.T @ t1) + t2;
    normal = np.array([0,0,1]).reshape((3,1))
    normal1 = R1 @ normal
    origin = np.zeros((3,1))
    origin1 = R1 @ origin + t1.reshape((3,1))
    d = (normal1.T @ origin1)[0,0]
    homo = R_1to2 + 1/d * t_1to2@normal1.T
    homo = cameraMatrix @ homo @ np.linalg.inv(cameraMatrix)

    homo = findHomo(init_img2, img)

    imgpts, _ = cv2.projectPoints(pts,
                              rvec, tvec, cameraMatrix, distCoeffs)
    # import pdb
    # pdb.set_trace()
    H, _ = cv2.findHomography(corners, imgpts)
    # print(HH)
    # H = np.linalg.inv(HH) @ H @ homo @ HH
    H = H @ homo @ HH

    warped = cv2.warpPerspective(init_img, H, (w,h))
    mask = cv2.warpPerspective(match_man_mask, H, (w, h))
    mask = mask > 0
    rst = img * (1-mask) + warped * mask
    return rst


def findHomo(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    corners1, ids1, _ = aruco.detectMarkers(gray1, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
    corners2, ids2, _ = aruco.detectMarkers(gray2, ARUCO_DICT, parameters=ARUCO_PARAMETERS)

    found_ids1 = []
    found_corners1 = []
    found_ids2 = []
    found_corners2 = []
    for i,id1 in enumerate(ids1):
        idx = np.where(ids2 == id1[0])
        if len(idx[0]) > 0:
            idx = idx[0][0]
            found_ids1.append(ids1[i])
            found_corners1.append(corners1[i])
            found_ids2.append(ids2[idx])
            found_corners2.append(corners2[idx])
    found_corners1 = np.array(found_corners1).reshape((-1,2))
    found_corners2 = np.array(found_corners2).reshape((-1,2))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners1_sub = cv2.cornerSubPix(gray1,np.array(found_corners1),(5,5),(-1,-1),criteria)
    corners2_sub = cv2.cornerSubPix(gray2,np.array(found_corners2),(5,5),(-1,-1),criteria)
    H, _ = cv2.findHomography(corners1_sub, corners2_sub,
                          method=cv2.RANSAC, ransacReprojThreshold=1)
    return H


def maskToConvex(mask):
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find the convex hull object for each contour
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)
    # Draw contours + hull results
    drawing = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (255,255,255)
        cv2.drawContours(drawing, contours, i, color)
        cv2.drawContours(drawing, hull_list, i, color, thickness=-1)
    return drawing

def run():
    cam = cv2.VideoCapture('video/VIDEO0123.mp4')
    frame_idx = 0
    T = np.eye(3)
    match_stack_man_cur = match_stack_man
    init = True
    init_img = None
    match_man_mask = None
    cropped_points = None
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('ar2.mp4', fourcc, 30.0, (384,216))
    while(cam.isOpened()):
        # Capturing each frame of our video stream
        ret, QueryImg = cam.read()
        if ret == True:
            # grayscale image
            # QueryImg = cv2.resize(QueryImg, (0,0), fx=0.2, fy=0.2)
            # cv2.imwrite("frames2/%06d.jpg" % frame_idx, QueryImg)
            QueryImg = cv2.imread("frames2/%06d.jpg" % frame_idx)
            frame_idx += 1
            gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)
        
            # Detect Aruco markers
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
    
            # Outline all of the markers detected in our image
            # QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, borderColor=(0, 0, 255))
            # Require 15 markers before drawing axis

            if init:
                
                h, w = QueryImg.shape[:2]
                mask = np.zeros((h,w), np.uint8)
                for corner in corners:
                    pts = corner.reshape((4,2)).astype(np.uint)
                    cv2.fillPoly(mask, pts = [pts], color=(255,255,255))
                gray = cv2.GaussianBlur(gray, (3,3), 3)
                mask2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                            cv2.THRESH_BINARY,5,2)
                _, mask2 = cv2.threshold(mask2,127,255,cv2.THRESH_BINARY)
                mask2 = 255 - mask2
                mask2 = np.bitwise_and(mask2, 255-mask)
                erosion_size = 1
                element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                                    (erosion_size, erosion_size))
                mask2[:10] = 0
                rst = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, element)
                erosion_size = 1
                element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                                    (erosion_size, erosion_size))
                rst = cv2.morphologyEx(rst, cv2.MORPH_DILATE, element)
                
                init_img = QueryImg * (rst > 0)[...,np.newaxis]
                match_man_mask = np.stack([rst, rst, rst], axis=2)
                y, x = np.where(rst > 0)
                minx, maxx = x.min(), x.max()
                miny, maxy = y.min(), y.max()
                cropped_points = np.array([minx, miny, maxx, maxy])
                HH, _ = cv2.findHomography(
                    np.array([[minx, miny],
                            [maxx, miny],
                            [maxx, maxy],
                            [minx, maxy]
                            ]),
                    corners[2]
                )
                # print(HH)
                init_img2 = QueryImg

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

                    if init:
                        init = False
                        init_rvec = rvec
                        init_tvec = tvec

                    QueryImg = draw(QueryImg, corners[idx], rvec, tvec, T, frame_idx, init_img, match_man_mask, init_rvec, init_tvec, HH, init_img2)
                    # T = rotating_matrix @ T


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