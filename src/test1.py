import cv2
import cv2.xfeatures2d as cv
from numpy import *
from matplotlib import pyplot as plt
import glob
from math import sqrt
import json


def MatchFeatures(img1, img2):
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    
    good1 = []
    try:
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good1.append([m])
    except ValueError:
        print("not enough matches")
        return False, [], []

    matches = bf.knnMatch(des2,des1, k=2)

    good2 = []
    try:
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good2.append([m])
    except ValueError:
        print("not enough matches")
        return False, [], []

    good=[]
    A = []
    B = []
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
                A.append([a2, b2])
                B.append([a1, b1])

    #result = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,[0,0,255],flags=2)
    #if (len(B) > 5) and (len(B) > 5):
    #    ok = True
    #else:
    #    ok = False
    #return ok, transpose(A), transpose(B)
    
    return len(good)


def rigid_transform_3D(A, B):
    A = mat(A)
    B = mat(B)
    assert len(A) == len(B)

    num_rows, num_cols = A.shape;

    if num_rows != 3:
        raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))

    [num_rows, num_cols] = B.shape;
    if num_rows != 3:
        raise Exception("matrix B is not 3xN, it is {}x{}".format(num_rows, num_cols))

    # find mean column wise
    centroid_A = mean(A, axis=1)
    centroid_B = mean(B, axis=1)

    # ensure centroids are 3x1 (necessary when A or B are 
    # numpy arrays instead of numpy matrices)
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - tile(centroid_A, (1, num_cols))
    Bm = B - tile(centroid_B, (1, num_cols))

    H = dot(Am, transpose(Bm))

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = linalg.svd(H)
    R = dot(Vt.T, U.T)

    # special reflection case
    if linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...\n");
        Vt[2,:] *= -1
        R = dot(Vt.T, U.T)

    t = -R*centroid_A + centroid_B

    return R, t


# set window size
cv2.namedWindow("img", cv2.WINDOW_NORMAL) 
cv2.resizeWindow('img', int(1080*2/3), 1080)

# default clear blank
default_img_name = "default.jpg"
img1 = cv2.imread(default_img_name, 0)
def_h = 720
img1 = cv2.resize(img1, (int(def_h*3/2), def_h))

# user uploaded images
folder = "photos/"
file_names = glob.glob(folder + '*.jpg')
#print(file_names)

for i in range(len(file_names)):
    img2 = cv2.imread(file_names[i], 0)
    #h, w = img1.shape
    img2 = cv2.resize(img2, (int(def_h*3/2), def_h))
    #combo = cv2.addWeighted(img1,0.5,img2,0.5,0)
    #ok, A, B = MatchFeatures(img1, img2)
    #if not ok:
    #    print("not ok.")
    #    #print(A)
    #    #print(B)
    #    continue
    #print(A)
    n = MatchFeatures(img1, img2)
    #_, n = A.shape
    #for i in range(n):
    #    cv2.line(combo, (int(A[0][i]), int(A[1][i])), (int(B[0][i]), int(B[1][i])), (0, 0, 255))
    """
    for i in range(n):
        A[0][i] = A[0][i] * height / f
        A[1][i] = A[1][i] * height / f
        B[0][i] = B[0][i] * height / f
        B[1][i] = B[1][i] * height / f
    Z = [repeat(height, n)]
    A = append(A, Z, axis=0)
    B = append(B, Z, axis=0)
    #print("A")
    #print(A)
    #print("B")
    #print(B)
    ret_R, ret_t = rigid_transform_3D(A, B)
    print("Recovered rotation")
    print(ret_R)
    print("")
    print("Recovered translation")
    print(ret_t)
    print("")
    """
    #cv2.imshow('img', img2)
    print("number of matches: " + str(n))
    #k = cv2.waitKey(0) & 0xff
