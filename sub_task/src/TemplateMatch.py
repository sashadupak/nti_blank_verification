import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob


tm = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
            cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

#cv2.namedWindow("Image", cv2.WINDOW_NORMAL) 

template = cv2.imread('../reference/logo2.jpg', 0)
template = cv2.resize(template, (200, 200))
(tH, tW) = template.shape[:2]

for imagePath in glob.glob('../data' + "/*.jpg"):
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (int(image.shape[1]/3), int(image.shape[0]/3)))
    #cv2.resizeWindow('Image', int(image.shape[1]/5), int(image.shape[0]/5))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    i_edges = cv2.Canny(gray, 50, 200)
    cv2.imshow("edges", i_edges)
    found = None

    # loop over the scales of the image
    for scale in np.linspace(0.1, 1.0, 50)[::-1]:
        resized = cv2.resize(template, (int(tW * scale), int(tH * scale)))
        r = float(resized.shape[1]) / template.shape[1]
        if resized.shape[0] > gray.shape[0] or resized.shape[1] > gray.shape[1]:
            continue

        t_edges = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(i_edges, t_edges, tm[3]) # 1-0.1, 3-0.2
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        if maxVal > 0.4:
            #print(maxVal)
            cv2.rectangle(image, (maxLoc[0], maxLoc[1]),
                (maxLoc[0] + resized.shape[0], maxLoc[1] + resized.shape[1]), (255, 0, 0), 2)
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)
            cv2.imshow("Image", image)
        cv2.imshow("template", t_edges)
        k = cv2.waitKey(0) & 0xff
        if k == ord('q'):
            exit()
    if found is None:
        print("not found")
    else:
        (_, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0]), int(maxLoc[1]))
        (endX, endY) = (int((maxLoc[0] + tW*r) ), int((maxLoc[1] + tH*r) ))
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.imshow("Image", image)
    k = cv2.waitKey(0) & 0xff
    if k == ord('q'):
        exit()

"""
img_rgb = cv.imread('../data/photo_2020-10-02_19-51-15.jpg')
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv.imread('../reference/logo1.jpg',0)
w, h = template.shape[::-1]
res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
cv.imwrite('res.png',img_rgb)
"""
