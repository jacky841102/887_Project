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
rotating_matrix = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0],
    [0, 0, 1]
])

def draw(img, corners, rvec, tvec, T, frame_idx, cropped_object, cropped_mask, HH):
    h, w = img.shape[:2]
    pts = np.array([[-1,0,1], [1,0,1], [1,0,0], [-1,0,0]]) * 1.5
    pts = np.hstack([pts, np.ones((4,1))])
    # pts[:,2] = pts[:,2] * 2
    
    pts = (T @ pts.T).T
    pts = pts[:,:3] / pts[:,3].reshape((-1,1))

    time1_ = 20
    t1 = frame_idx % time1_
    v = 0.1
    g = 2 * v / time1_
    dz = v * t1 - 0.5 * g * t1 * t1

    r = 4
    time2_ = 150
    t2 = frame_idx % time2_
    omega = 2 * np.pi / time2_
    dx = -np.cos(omega * t2) * r
    dy = np.sin(omega * t2) * r

    pts[:,0] = pts[:,0] - 2.5
    # pts[:,0] = pts[:,0] + dx
    # pts[:,1] = pts[:,1] + dy
    # pts[:,2] = pts[:,2] + dz

    imgpts, _ = cv2.projectPoints(pts,
                              rvec, tvec, cameraMatrix, distCoeffs)
    H, _ = cv2.findHomography(corners, imgpts)
    H = H @ HH

    warped = cv2.warpPerspective(cropped_object, H, (w,h))
    mask = cv2.warpPerspective(cropped_mask, H, (w, h))
    mask = mask > 127
    rst = img * (1-mask) + warped * mask
    return rst


def findHomo(corners1, ids1, corners2, ids2):
    # gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # corners1, ids1, _ = aruco.detectMarkers(gray1, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
    # corners2, ids2, _ = aruco.detectMarkers(gray2, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
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
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # corners1_sub = cv2.cornerSubPix(gray1,np.array(found_corners1),(5,5),(-1,-1),criteria)
    # corners2_sub = cv2.cornerSubPix(gray2,np.array(found_corners2),(5,5),(-1,-1),criteria)
    H, _ = cv2.findHomography(found_corners1, found_corners2,
                          method=cv2.RANSAC, ransacReprojThreshold=1)
    return H


def cropObject(img, corners):
    h, w = img.shape[:2]
    mask = np.zeros((h,w), np.uint8)
    
    # mask for markers
    for corner in corners:
        pts = corner.reshape((4,2)).astype(np.uint)
        cv2.fillPoly(mask, pts = [pts], color=(255,255,255))

    # denoise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 3)

    # mask for object
    mask2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,5,2)
    _, mask2 = cv2.threshold(mask2,127,255,cv2.THRESH_BINARY)

    # combine masks
    mask2 = 255 - mask2
    mask2 = np.bitwise_and(mask2, 255-mask)

    # remove small dot noises
    erosion_size = 1
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                        (erosion_size, erosion_size))

    # heuristically mask out boundary pixels
    mask2[:30] = 0

    # dilate boundary
    rst = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, element)
    erosion_size = 1
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                        (erosion_size, erosion_size))
    rst = cv2.morphologyEx(rst, cv2.MORPH_DILATE, element)

    # contour mask
    mask = maskToConvex(rst)

    # cropped region
    y, x = np.where(rst > 0)
    minx, maxx = x.min(), x.max()
    miny, maxy = y.min(), y.max()

    cropped_object = img[miny:maxy+1,minx:maxx+1]
    cropped_mask = mask[miny:maxy+1,minx:maxx+1]

    # homography from cropped img to origin place
    H_obj2ori = np.array([[1,0,minx],[0,1,miny],[0,0,1]])
    return cropped_object, cropped_mask, H_obj2ori, np.array([minx, miny, maxx, maxy])


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

def control(key, T, R):
    if key == 0 or key == ord('w'):
        # up
        T = np.array([[1,0,0,0],[0,1,0,0.3],[0,0,1,0],[0,0,0,1]]) @ T
        # print("up")
    elif key == 1 or key == ord('s'):
        # down
        T = np.array([[1,0,0,0],[0,1,0,-0.3],[0,0,1,0],[0,0,0,1]]) @ T
        # print("down")
    elif key == 2 or key == ord('a'):
        # left
        T = np.array([[1,0,0,-0.3],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) @ T
        # print("left")
    elif key == 3 or key == ord('d'):
        # right
        T = np.array([[1,0,0,0.3],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) @ T
    elif key == ord('z'):
        M = np.eye(4)
        M[:3,:3] = rotating_matrix
        R = M @ R
    elif key == ord('x'):
        M = np.eye(4)
        M[:3,:3] = rotating_matrix.T
        R = M @ R
    return T, R

def run():
    cam = cv2.VideoCapture('video/VIDEO0124.mp4')
    frame_idx = 0
    T = np.eye(4)
    R = np.eye(4)
    match_stack_man_cur = match_stack_man
    init = True
    cropped_points = None
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('ar3.mp4', fourcc, 30.0, (384,216))
    while(cam.isOpened()):
        # Capturing each frame of our video stream
        ret, QueryImg = cam.read()
        if ret == True:
            # grayscale image
            QueryImg = cv2.resize(QueryImg, (0,0), fx=0.2, fy=0.2)
            # cv2.imwrite("frames3/%06d.jpg" % frame_idx, QueryImg)
            # QueryImg = cv2.imread("frames3/%06d.jpg" % frame_idx)
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

                    if init:
                        init = False
                        cropped_object, cropped_mask, H_obj2ori, cropped_points = cropObject(QueryImg, corners)
                        HH, _ = cv2.findHomography(
                            np.array([[cropped_points[0], cropped_points[1]],
                                    [cropped_points[2], cropped_points[1]],
                                    [cropped_points[2], cropped_points[3]],
                                    [cropped_points[0], cropped_points[3]]
                                    ]),
                            corners[1]
                        )
                        HH = HH @ H_obj2ori
                        init_rvec = rvec
                        init_tvec = tvec
                        init_corners, init_ids = corners, ids

                    H = findHomo(init_corners, init_ids, corners, ids)
                    H = H @ HH
                    QueryImg = draw(QueryImg, corners[idx], rvec, tvec, T @ R, frame_idx, cropped_object, cropped_mask, H)
                    # T = rotating_matrix @ T


            cv2.imshow('QueryImage', QueryImg.astype(np.uint8))
            out.write(QueryImg.astype(np.uint8))
    
        # Exit at the end of the video on the 'q' keypress
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        T, R = control(key, T, R)
    cv2.destroyAllWindows()
    cam.release()
    out.release()


if __name__ == "__main__":
    run()