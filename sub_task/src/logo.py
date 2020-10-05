import cv2
file_name = "../reference/logo7.jpg"

src = cv2.imread(file_name)
resized = cv2.resize(src, (500, 500))
edges = cv2.Canny(resized, 50, 200)

cv2.imshow('result', edges)
cv2.waitKey(0)