import cv2
import cv2.xfeatures2d as cv # only for the new version
import numpy as np
from matplotlib import pyplot as plt
import glob
from math import sqrt
import json
import csv
import os


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
        ref1_id1=i[0].queryIdx 
        img2_id1=i[0].trainIdx
        (x1,y1)=kp1[ref1_id1].pt
        (x2,y2)=kp2[img2_id1].pt

        for j in good2:
            ref1_id2=j[0].queryIdx
            img2_id2=j[0].trainIdx

            (a1,b1)=kp2[ref1_id2].pt
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
        #print("Correct blank")
        return True
    else:
        #print("Wrong blank")
        return False


def transform(img2, good, kp1, kp2):
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img2.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    cropped = cv2.warpPerspective(img2, M, (w, h))

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    return cropped, img2


def confirm_sign(sign, min_sign_area):
    edges = cv2.Canny(sign,100,200)
    ret, thresh = cv2.threshold(edges, 127, 255, 0)
    blur = cv2.blur(thresh,(7,7))
    _, contours, hierarchy = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(sign, contours, -1, (0,255,0), 1)
    if len(contours) > 0:
        contour_area = max([cv2.contourArea(cnt) for cnt in contours])
    else:
        contour_area = 0
    #print("sign area: " + str(contour_area))
    #cv2.imshow('sign', sign)
    if contour_area > min_sign_area:
        return True, contour_area
    else:
        return False, contour_area


def main():
    # set window size
    #cv2.namedWindow("window", cv2.WINDOW_NORMAL) 
    #cv2.resizeWindow('window', int(640*2/3), 640)

    os.chdir("../")
    fh = open('results.csv', 'w')
    writer = csv.writer(fh)
    writer.writerow(["File name", "Blank type", "number of matches", "correct blank", "area of sign", "confirmed"])

    # main variables
    def_h = 750 # image resolution (y axis) - dependent (TODO independent)
    min_match = 50 # min number of matches, correct = 70-364, wrong = 0-33
    min_sign_area = [500, 750, 550] # min area of sign, correct = 599-899/1057-1142/752-897, wrong = 0-392/0-491/0-366

    # load default clear blanks
    ref_folder = 'reference/'
    ref_names = ['default.jpg', 'default2.jpg', 'default3.jpg']
    # sign position, format [y1, y2, x1, x2] 
    sp = [[650, 690, 350, 400], [615, 645, 135, 185], [660, 690, 190, 250]]  
    #TODO sp*h_def/750, also for min match and sign area
    ref = {}
    for i in range(len(ref_names)):
        ref[i] = cv2.imread(ref_folder + ref_names[i], 0)
        ref[i] = cv2.resize(ref[i], (int(def_h*2/3), def_h))

    # user uploaded images
    data_folder = "photos/"
    file_names = glob.glob(data_folder + '*.jpg')
    #print(file_names)

    for j in range(len(file_names)):
        #print("File name: " + file_names[i])
        img = cv2.imread(file_names[j], 0)
        img = cv2.resize(img, (int(def_h*2/3), def_h))
        
        match_no = []
        for i in range(len(ref)):
            des1, des2, kp1, kp2 = SIFT(img, ref[i])
            good = BFMatch(des1, des2, kp1, kp2)
            match_no.append(len(good))
        i = match_no.index(max(match_no))
        des1, des2, kp1, kp2 = SIFT(img, ref[i]) # not efficient TODO rewrite
        good = BFMatch(des1, des2, kp1, kp2) # the same
        ok = verify_blank(good, min_match)
        confirmed = 0
        sign_area = 0
        if ok:
            cropped, original = transform(img, good, kp1, kp2)
            sign = cropped[sp[i][0]:sp[i][1], sp[i][2]:sp[i][3]] # attention: format [y1:y2, x1:x2]  
            # TODO inshot = check_shot(cropped) - in func: threshhold and calculate area of blank (should be more than 90%)
            confirmed, sign_area = confirm_sign(sign, min_sign_area[i])
            cv2.rectangle(cropped, (sp[i][2], sp[i][0]), (sp[i][3], sp[i][1]), (0, 0, 255), 2)
            #cv2.imshow('window', cropped)
        writer.writerow([file_names[j].split("\\")[1], str(i), str(len(good)), str(int(ok)), str(sign_area), str(int(confirmed))])
        #k = cv2.waitKey(0) & 0xff
        #if k == ord('q'):
        #    exit()


if __name__ == "__main__":
    main()
