import cv2
import cv2.xfeatures2d as cv # only for a new version
import numpy as np
from matplotlib import pyplot as plt
import glob
from math import sqrt
import json
import csv
import os
from BlankType import BlankType
from BlankStatus import BlankStatus


def Match(des1, des2, kp1, kp2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    
    good1 = []
    try:
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good1.append(m)
    except ValueError:
        print("not enough matches")
        return None

    matches = bf.knnMatch(des2,des1, k=2)

    good2 = []
    try:
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good2.append(m)
    except ValueError:
        print("not enough matches")
        return None

    good=[]
    for i in good1:
        ref1_id1=i.queryIdx 
        img2_id1=i.trainIdx
        (x1,y1)=kp1[ref1_id1].pt
        (x2,y2)=kp2[img2_id1].pt

        for j in good2:
            ref1_id2=j.queryIdx
            img2_id2=j.trainIdx

            (a1,b1)=kp2[ref1_id2].pt
            (a2,b2)=kp1[img2_id2].pt

            if (a1 == x2 and b1 == y2) and (a2 == x1 and b2 == y1):
                good.append(i)
    return good


def verify_blank(matches, min_match):
    if len(matches) > min_match:
        #print("Correct blank")
        return BlankStatus.CORRECT
    else:
        #print("Wrong blank")
        return BlankStatus.WRONG


def transform(img, matches, kp1, kp2):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    cropped = cv2.warpPerspective(img, M, (w, h))
    return cropped


def confirm_sign(sign, min_sign_area):
    edges = cv2.Canny(sign,100,200)
    ret, thresh = cv2.threshold(edges, 127, 255, 0)
    blur = cv2.blur(thresh,(7,7))
    _, contours, hierarchy = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(sign, contours, -1, (0,255,0), 1)
    if len(contours) > 0:
        contour_area = max([cv2.contourArea(cnt) for cnt in contours])
    else:
        contour_area = 0
    #print("sign area: " + str(contour_area))
    #cv2.imshow('sign', sign)
    if contour_area > min_sign_area:
        return BlankStatus.CONFIRMED, contour_area
    else:
        return BlankStatus.CORRECT, contour_area


def main():
    # create sift
    sift = cv.SIFT_create()

    # set window size
    cv2.namedWindow("window", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow('window', int(640*2/3), 640)

    # create log file
    os.chdir("../")
    fh = open('results1.csv', 'w')
    writer = csv.writer(fh)
    writer.writerow(["File name", "number of matches", "area of sign", "Blank type"])

    # main variables
    def_h = 750 # image resolution (y axis) - dependent (TODO make independent) - don't set less than 700px 
    min_match = 50 # = func(def_h) # min number of matches, correct = 70-364, wrong = 0-33
    min_sign_area = [500, 750, 550] # min area of sign, correct = 599-899/1057-1142/752-897, wrong = 0-392/0-491/0-366
    # TODO minimaze sign frame to minimaze cnt area of empty blank -> sp
    # min_sign_area should be the same
    # min sign area about 30%

    # load default clear blanks
    ref_folder = 'reference/'
    ref_names = ['default.jpg', 'default2.jpg', 'default3.jpg']
    blank_types = [BlankType.GENERAL, BlankType.CHILD18, BlankType.CHILD14]
    ref = {}
    kp2 = {}
    des2 = {}
    for ref_name in ref_names:
        ref[ref_name] = cv2.imread(ref_folder + ref_name, 0)
        ref[ref_name] = cv2.resize(ref[ref_name], (int(def_h*2/3), def_h))
        kp2[ref_name], des2[ref_name] = sift.detectAndCompute(ref[ref_name], None)

    # sign position, format [y1, y2, x1, x2] 
    sp = [[650, 690, 350, 400], [615, 645, 135, 185], [660, 690, 190, 250]]
    #sp = [int(i * def_h/750) for i in sp] TODO
    #TODO sp*def_h/750, also for min match and sign area 

    # user uploaded images
    data_folder = "photos/"
    file_names = glob.glob(data_folder + '*.jpg')

    for file_name in file_names:
        img = cv2.imread(file_name, 0)
        img = cv2.resize(img, (int(def_h*2/3), def_h))
        kp1, des1 = sift.detectAndCompute(img, None)
        
        match_count = 0
        matches_list = []
        for ref_name in ref_names:
            matches = Match(des1, des2[ref_name], kp1, kp2[ref_name])
            if matches is None:
                continue
            matches_list.append(matches)
            if len(matches) > match_count:
                match_count = len(matches)
                ind = ref_names.index(ref_name)

        sign_area = 0
        if len(matches_list) > 0:
            blank_type = blank_types[ind]
            matches = matches_list[ind]
            ref_name = ref_names[ind]

            blank_status = verify_blank(matches, min_match)
            if blank_status == BlankStatus.CORRECT:
                cropped = transform(img, matches, kp1, kp2[ref_name])
                sign = cropped[sp[ind][0]:sp[ind][1], sp[ind][2]:sp[ind][3]] # attention: format [y1:y2, x1:x2]  
                # TODO inshot = check_shot(cropped) - in func: threshhold and calculate area of blank (should be more than 90%)
                blank_status, sign_area = confirm_sign(sign, min_sign_area[ind])
                cv2.rectangle(cropped, (sp[ind][2], sp[ind][0]), (sp[ind][3], sp[ind][1]), (0, 0, 255), 2)
                cv2.imshow('window', cropped)
        else:
            matches = []
            blank_type = BlankType.UNDEFINED
        writer.writerow([file_name.split("\\")[1], str(len(matches)), str(sign_area), str(blank_type)])
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            exit()


if __name__ == "__main__":
    main()
