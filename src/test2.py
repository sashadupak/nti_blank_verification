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


def Fast(img1):
    fast = cv2.FastFeatureDetector()
    kp = fast.detect(img1,None)
    ### TODO
    return


def BRIEF(img1, img2):
    star = cv2.FeatureDetector_create("STAR")
    brief = cv2.DescriptorExtractor_create("BRIEF")
    kp1 = star.detect(img1,None)
    kp2 = star.detect(img2,None)
    kp1, des1 = brief.compute(img1, kp1)
    kp2, des2 = brief.compute(img2, kp2)
    return des1, des2, kp1, kp2


def ORB(img1, img2):
    orb = cv2.ORB()
    kp1 = orb.detect(img1,None)
    kp1, des1 = orb.compute(img1, kp1)
    kp2, des2 = orb.compute(img2, kp2)
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
                good.append(i)
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
        print("Number of matches: " + str(len(good)))
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


def_h = 750 # image resolution (y axis)
min_match = 50 # min number of matches

# set window size
cv2.namedWindow("window", cv2.WINDOW_NORMAL) 
cv2.resizeWindow('window', int(750*2/3), 750)

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
    good = FlannMatch(des1, des2)
    ok = verify_blank(good, min_match)
    print("(" + file_names[i] + ")")
    if ok:
        cropped, original = transform(img2, good, kp1, kp2)
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
