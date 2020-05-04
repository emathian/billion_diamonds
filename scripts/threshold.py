import cv2
import os
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu
import numpy as np
file_path = "C:/Users/ungestef/Desktop/cv-pipeline-fixed_bounding_boxes/images/cut_dataset/"
images = os.listdir(file_path)

for image in images:

    print(image)
    img = cv2.imread(file_path+image,cv2.IMREAD_GRAYSCALE)
#    img = cv2.equalizeHist(img)
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    plt.plot(hist)
    plt.xlim([0,256])

    cv2.imshow("orig", img)
    img = cv2.medianBlur(img,9)

    ret,thresh1 = cv2.threshold(img,50,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)
    thresholds = threshold_multiotsu(img, classes=5)
    print(thresholds)
    otsu = np.uint8(np.digitize(img, bins=thresholds))
    otsu[otsu == 0] = 0
    otsu[otsu == 1] = 49
    otsu[otsu == 2] = 99
    otsu[otsu == 3] = 255
    otsu[otsu == 4] = 255
    otsu[otsu == 5] = 255
    otsu[otsu == 6] = 255
    cv2.imshow("thresh",thresh1)
    cv2.imshow("mean",th2)
    cv2.imshow("gauss",th3)
    cv2.imshow("otsu", otsu)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
plt.show()

