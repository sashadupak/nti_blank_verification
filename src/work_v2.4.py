#!/usr/bin/env python
# coding: utf-8
import cv2
import cv2.xfeatures2d as cv # only for a new version
import numpy as np
from matplotlib import pyplot as plt
import glob
from math import sqrt
import json
import csv
import os
import time
import traceback
import sys
from BlankType import BlankType
from BlankStatus import BlankStatus


class Verificator:

    def __init__(self):
        self.output_file_name = 'results1.csv'
        self.ref_folder = 'reference/'
        self.ref_names = ['default.jpg', 'default2.jpg', 'default3.jpg']
        self.blank_types = [BlankType.GENERAL, BlankType.CHILD18, BlankType.CHILD14] # corresponding to names order
        self.data_folder = 'data/ipad/'


    def match_features(self, des1, des2, kp1, kp2):
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


    def verify_blank(self, matches, min_match):
        if len(matches) > min_match:
            #print("Correct blank")
            return BlankStatus.CORRECT
        else:
            #print("Wrong blank")
            return BlankStatus.WRONG


    def transform(self, img, matches, kp1, kp2):
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        cropped = cv2.warpPerspective(img, M, (w, h))
        corners = [np.int32(dst)]
        return cropped, corners


    def confirm_sign(self, sign, min_sign_area):
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


    def check_shot(self, img, bounds):
        ret, thresh = cv2.threshold(img, 127, 255, 0) # TODO area should be more than 90%
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            max_area = 0
            max_cnt = None
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > max_area:
                    max_area = area
                    max_cnt = cnt
            if max_cnt is not None:
                #cv2.drawContours(thresh, [max_cnt], -1, (0,255,0), 1)
                rect = cv2.minAreaRect(max_cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.imshow('thresh', thresh)
                for x, y in box:
                    if (x < bounds[0]) or (x > bounds[2]):
                        return BlankStatus.OUTOFSHOT
                    if (y < bounds[1]) or (y > bounds[3]):
                        return BlankStatus.OUTOFSHOT
                #print(corners)
                return BlankStatus.CORRECT
            else:
                return BlankStatus.OUTOFSHOT
        else:
            return BlankStatus.OUTOFSHOT


    def main(self):
        # create sift
        sift = cv.SIFT_create()

        # set window size
        cv2.namedWindow("window", cv2.WINDOW_NORMAL) 
        cv2.resizeWindow('window', int(640*2/3), 640)

        # create log file
        writer = csv.writer(self.fh)
        writer.writerow(["File name", "number of matches", "area of sign", "Blank type", "Blank status", "OK?"])

        # main variables
        def_h = 750 # image resolution (y axis) - dependent (TODO independent) - don't set less than 700px 
        min_match = 50 # min number of matches (correct = 70-364, wrong = 0-33) TODO func(def_h) without user correction
        min_sign_area = [500, 750, 550] # min area of sign, correct = 599-899/1057-1142/752-897, wrong = 0-392/0-491/0-366
        # TODO minimaze sign frame to minimaze cnt area of empty blank -> sp
        # min_sign_area should be the same
        # min sign area should be about 30% - make func(def_h)

        # load default clear blanks
        ref = {}
        kp2 = {}
        des2 = {}
        for ref_name in self.ref_names:
            ref[ref_name] = cv2.imread(self.ref_folder + ref_name, 0)
            ref[ref_name] = cv2.resize(ref[ref_name], (int(def_h*2/3), def_h))
            kp2[ref_name], des2[ref_name] = sift.detectAndCompute(ref[ref_name], None)


        # text bounds
        h, w = ref[self.ref_names[0]].shape
        bounds = [-20, -30, w+20, h+50] # format [x_l, y_up, x_r, y_down]

        # sign position, format [y1, y2, x1, x2] 
        sp = [[650, 690, 350, 400], [615, 645, 135, 185], [660, 690, 190, 250]]
        # TODO sp = [int(i * def_h/750) for i in sp]

        # user uploaded images
        file_names = glob.glob(self.data_folder + '*.jpg')
        self.blank_count = 0

        for file_name in file_names:
            self.blank_count += 1
            img = cv2.imread(file_name, 0)
            img = cv2.resize(img, (int(def_h*2/3), def_h))
            kp1, des1 = sift.detectAndCompute(img, None)
            
            match_count = 0
            matches_list = []
            for ref_name in self.ref_names:
                matches = self.match_features(des1, des2[ref_name], kp1, kp2[ref_name])
                if matches is None:
                    continue
                matches_list.append(matches)
                if len(matches) > match_count:
                    match_count = len(matches)
                    ind = self.ref_names.index(ref_name)

            sign_area = 0
            if len(matches_list) > 0:
                blank_type = self.blank_types[ind]
                matches = matches_list[ind]
                ref_name = self.ref_names[ind]

                blank_status = self.verify_blank(matches, min_match)
                if blank_status == BlankStatus.CORRECT:
                    cropped, corners = self.transform(img, matches, kp1, kp2[ref_name])
                    sign = cropped[sp[ind][0]:sp[ind][1], sp[ind][2]:sp[ind][3]] # attention: format [y1:y2, x1:x2]  
                    blank_status = self.check_shot(cropped, bounds)
                    #print(blank_status)
                    if blank_status != BlankStatus.OUTOFSHOT:
                        blank_status, sign_area = self.confirm_sign(sign, min_sign_area[ind])
                        cv2.rectangle(cropped, (sp[ind][2], sp[ind][0]), (sp[ind][3], sp[ind][1]), (0, 0, 255), 2)
                    cv2.rectangle(cropped, (bounds[0], bounds[1]), (bounds[2], bounds[3]), (0, 0, 255), 3)
                    cv2.imshow('window', cropped)
            else:
                matches = []
                blank_type = BlankType.UNDEFINED
            if blank_status == BlankStatus.CONFIRMED:
                ok = True
            else:
                ok = False
            writer.writerow([file_name.split("\\")[1], str(len(matches)), str(sign_area), str(blank_type.value), str(blank_status.value), str(ok)])
            k = cv2.waitKey(1) & 0xff
            if k == ord('q'):
                exit()


if __name__ == "__main__":
    verificator = None

    try:
        start_time = time.time()
        verificator = Verificator()

        os.chdir("../")
        verificator.fh = open(verificator.output_file_name, 'w')
        verificator.main()
    except Exception:
        traceback.print_exc(file=sys.stdout)
    else:
        elapsed_time = time.time() - start_time
        print("Elapsed time: %.2f s" % elapsed_time)
        print("Estimated: %.2f seconds per sample" % float(elapsed_time / verificator.blank_count))
    finally:
        verificator.fh.close()
        del verificator
