import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

file_path = "C:/Users/ungestef/Desktop/cv-pipeline-fixed_bounding_boxes/images/orig_dataset/"
images = os.listdir(file_path)
#print(images)
#for image in images:
#    print(image)
#    img = cv2.imread(file_path+image,cv2.IMREAD_GRAYSCALE)
#    hist = cv2.calcHist([img],[0],None,[256],[0,256])
#    plt.plot(hist)
#    plt.xlim([0,256])
#plt.show()

img = cv2.imread(file_path+images[0],cv2.IMREAD_GRAYSCALE)
pts_list = [
    # x1, y1, x2, y2
    # segment 1
    [560, 560, 1398, 791],
    # segment 2
    [1580, 549, 2456, 779],
    # segment 3
    [2637, 527, 3488, 761],
    # segment 4
    [500, 988, 1379, 1232],
    # segment 5
    [1573, 975, 2460, 1219],
    # segment 6
    [2661, 963, 3544, 1204],
    # segment 7
    [435, 1454, 1360, 1713],
    # segment 8
    [1550, 1430, 2487, 1713],
    # segment 9
    [2683, 1420, 3608, 1690],
    # segment 10
    [366, 1943, 1316, 2261],
    # segment 11
    [1529, 1937, 2498, 2236],
    # segment 12
    [2709, 1927, 3678, 2229]
]

# create mask for each segment
mask = np.zeros(img.shape[:2], np.uint8)

for i, pts in enumerate(pts_list):
        #self.segments.append(Segment(pts[0], pts[1], pts[2] - pts[0],
        #                                pts[3] - pts[1]))

    print(pts[0],pts[1], pts[2],pts[3])
    #y oben, y unten, x links, x rechts
    mask[pts[1]:pts[3], pts[0]:pts[2]] = 255
    masked_img = cv2.bitwise_and(img,img,mask = mask)
    hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])

    plt.plot(hist_mask)
    plt.xlim([0,256])
#plt.subplot(221), plt.imshow(img, 'gray')
#plt.subplot(222), plt.imshow(mask,'gray')
#plt.subplot(223), plt.imshow(masked_img, 'gray')
#plt.subplot(224), plt.plot(hist_mask)
plt.show()
