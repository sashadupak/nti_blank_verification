from work_2 import SIFT, BFMatch
import cv2
import os
import time
import glob

# set window size
#cv2.namedWindow("window", cv2.WINDOW_NORMAL) 
#cv2.resizeWindow('window', int(640*2/3), 640)

os.chdir("../")
fh = open("output1.txt", 'w')
fh.write("pixels \t number_of_matches \t time_elapsed \n")

# load reference files
ref_folder = 'reference/'
ref_names = ['default.jpg', 'default2.jpg', 'default3.jpg']
ref_raw = {}
for ref_name in ref_names:
    ref_raw[ref_name] = cv2.imread(ref_folder + ref_name, 0)

# user uploaded images
data_folder = "photos/"
file_names = glob.glob(data_folder + '*.jpg')

for def_h in range(500, 1500, 100):
    # resize reference blanks
    ref = {}
    for ref_name in ref_names:
        ref[ref_name] = cv2.resize(ref_raw[ref_name], (int(def_h*2/3), def_h))

    # open data files
    for file_name in file_names:
        img = cv2.imread(file_name, 0)
        img = cv2.resize(img, (int(def_h*2/3), def_h))

        # process
        for ref_name in ref_names:
            start_time = time.time()
            des1, des2, kp1, kp2 = SIFT(img, ref[ref_name])
            good = BFMatch(des1, des2, kp1, kp2)
            duration_time = time.time() - start_time

            """
            # show difference
            combo = cv2.addWeighted(img1,0.5,img2,0.5,0)
            for i in good:
                img1_id1=i.queryIdx 
                img2_id1=i.trainIdx
                (x1,y1)=kp1[img1_id1].pt
                (x2,y2)=kp2[img2_id1].pt
                cv2.line(combo, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255))
            """

            fh.write(str(def_h) + "\t" + str(len(good)) + "\t" + str(duration_time) + "\n")
            #print("Number of matches: " + str(len(good)))
            #cv2.imshow('window', combo)
            #k = cv2.waitKey(0) & 0xff
            #if k == ord('q'):
            #    exit()
