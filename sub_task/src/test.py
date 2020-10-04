import cv2
import cv2.xfeatures2d as cv # only for a new version
import numpy as np
import glob
import os


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


# create sift
sift = cv.SIFT_create()

# set window size
cv2.namedWindow("window", cv2.WINDOW_NORMAL) 
#cv2.resizeWindow('window', int(640*2/3), 640)

def_h = 1500
min_match = 50

os.chdir("../")
ref_folder = 'reference/logo1.jpg'
ref = cv2.imread(ref_folder, 0)
#ref = cv2.resize(ref, (int(def_h*2/3), def_h))
kp2, des2 = sift.detectAndCompute(ref, None)

data_folder = "data/"
file_names = glob.glob(data_folder + '*.jpg')

for file_name in file_names:
    img = cv2.imread(file_name, 0)
    h, w = img.shape
    ref = cv2.resize(ref, (w, h))
    #img = cv2.resize(img, (int(def_h*2/3), def_h))
    kp1, des1 = sift.detectAndCompute(img, None)
    matches = Match(des1, des2, kp1, kp2)
    if matches is None:
        continue

    """
    combo = cv2.addWeighted(img,0.5,ref,0.5,0)
    for i in matches:
        img1_id1=i.queryIdx 
        img2_id1=i.trainIdx
        (x1,y1)=kp1[img1_id1].pt
        (x2,y2)=kp2[img2_id1].pt
        cv2.line(combo, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255))
    """

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    combo = cv2.drawMatches(img,kp1,ref,kp2,matches,None,**draw_params)

    print(len(matches))
    cv2.imshow("window", combo)

    k = cv2.waitKey(0) & 0xff
    if k == ord('q'):
        exit()