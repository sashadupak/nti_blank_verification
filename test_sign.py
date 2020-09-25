import cv2
import cv2.xfeatures2d as cv # only for the new version
import numpy as np
from matplotlib import pyplot as plt
import glob
from math import sqrt
import json


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


# load sign examples
folder = "sign_example/"
file_names = glob.glob(folder + '*.png')
#print(file_names)

for i in range(len(file_names)):
    sign = cv2.imread(file_names[i], 0)
    sign = cv2.resize(sign, (50, 40))
    
    confirmed = confirm_sign(sign)
    if confirmed:
        print("Sign confirmed")
    else:
        print("Bad sign")

    cv2.imshow('sign', sign)

    k = cv2.waitKey(0) & 0xff
    if k == ord('q'):
        exit()
