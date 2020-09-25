import cv2
import cv2.xfeatures2d as cv # only for the new version
import numpy as np
from matplotlib import pyplot as plt
import glob
from math import sqrt
import json


def SIFT(img1, img2):
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    return des1, des2, kp1, kp2


def SURF(img1, img2):
    surf = cv.SURF_create()
    kp1, des1 = surf.detectAndCompute(img1, None)
    kp2, des2 = surf.detectAndCompute(img2, None)
    return des1, des2, kp1, kp2


def BFMatch(des1, des2, kp1, kp2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    
    good1 = []
    try:
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good1.append([m])
    except ValueError:
        print("not enough matches")
        return None

    matches = bf.knnMatch(des2,des1, k=2)

    good2 = []
    try:
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good2.append([m])
    except ValueError:
        print("not enough matches")
        return None

    good=[]
    for i in good1:
        img1_id1=i[0].queryIdx 
        img2_id1=i[0].trainIdx
        (x1,y1)=kp1[img1_id1].pt
        (x2,y2)=kp2[img2_id1].pt

        for j in good2:
            img1_id2=j[0].queryIdx
            img2_id2=j[0].trainIdx

            (a1,b1)=kp2[img1_id2].pt
            (a2,b2)=kp1[img2_id2].pt

            if (a1 == x2 and b1 == y2) and (a2 == x1 and b2 == y1):
                good.append(i[0])
    return good


def FlannMatch(des1, des2):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    return good


def verify_blank(good, min_match):
    if len(good)>min_match:
        print("Correct black. Number of matches: " + str(len(good)))
        return True
    else:
        print("Wrong blank! Not enough matches are found - %d/%d" % (len(good),min_match))
        return False


def transform(img2, good, kp1, kp2):
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    cropped = cv2.warpPerspective(img2, M, (w, h))

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    return cropped, img2


def confirm_sign(sign):
    edges = cv2.Canny(sign,100,200)
    ret, thresh = cv2.threshold(edges, 127, 255, 0)
    blur = cv2.blur(thresh,(7,7))
    _, contours, hierarchy = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(sign, contours, -1, (0,255,0), 1)
    #length = 0
    #for cnt in contours:
    #    length += cv2.arcLength(cnt, False)
    if len(contours) > 0:
        contour_area = max([cv2.contourArea(cnt) for cnt in contours])
    else:
        contour_area = 0
    print("sign area: " + str(contour_area))
    cv2.imshow('sign', sign)
    if contour_area > 500:
        return True
    else:
        return False


def_h = 750 # image resolution (y axis)
min_match = 50 # min number of matches

# set window size
cv2.namedWindow("window", cv2.WINDOW_NORMAL) 
cv2.resizeWindow('window', int(640*2/3), 640)

# load default clear blank
default_img_name = "default.jpg"
img1 = cv2.imread(default_img_name, 0)
img1 = cv2.resize(img1, (int(def_h*2/3), def_h))

# user uploaded images
folder = "photos/"
file_names = glob.glob(folder + '*.jpg')
#print(file_names)

for i in range(len(file_names)):
    img2 = cv2.imread(file_names[i], 0)
    img2 = cv2.resize(img2, (int(def_h*2/3), def_h))
    
    des1, des2, kp1, kp2 = SIFT(img2, img1)
    good = BFMatch(des1, des2, kp1, kp2)
    ok = verify_blank(good, min_match)
    print("(" + file_names[i] + ")")
    if ok:
        cropped, original = transform(img2, good, kp1, kp2)
        sign = cropped[650:690, 350:400] # attention: format [y1:y2, x1:x2]  
        confirmed = confirm_sign(sign)
        if confirmed:
            print("Sign confirmed")
        else:
            print("Bad sign")
        cv2.rectangle(cropped, (350, 650), (400, 690), (0, 0, 255), 2)
        cv2.imshow('window', cropped)
    else:
        cv2.imshow('window', img2)

    """
    combo = cv2.addWeighted(img1,0.5,img2,0.5,0)
    for i in range(len(good)):
        img1_id1=i[0].queryIdx 
        img2_id1=i[0].trainIdx
        (x1,y1)=kp1[img1_id1].pt
        (x2,y2)=kp2[img2_id1].pt
        cv2.line(combo, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255))
    """

    k = cv2.waitKey(0) & 0xff
    if k == ord('q'):
        exit()
