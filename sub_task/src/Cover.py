from TemplateMatchOne import main
import glob
import cv2

for imagePath in glob.glob('../data' + "/*.jpg"):
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (int(image.shape[0]/3), int(image.shape[1]/3)))
    cv2.imshow("im", image)
    if cv2.waitKey(1) == ord('q'):
        exit()
    main({'input': '../data/photo_2020-10-02_19-51-15.jpg'})
