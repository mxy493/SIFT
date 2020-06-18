# -*- coding: UTF-8 -*-
# author: mxy

import cv2

# 读入两张图片
img1 = cv2.imread('img/img3.jpg')
img2 = cv2.imread('img/img4.jpg')
rows, cols = img2.shape[:2]
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.6 * n.distance:
        good.append([m])
img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[:40], None, (255, 0, 0), flags=2)

# 获取img1左上角坐标
x0 = int(min(kp1[x[0].queryIdx].pt[0] for x in good))
y0 = int(min(kp1[x[0].queryIdx].pt[1] for x in good))
x1 = int(max(kp1[x[0].queryIdx].pt[0] for x in good))
y1 = int(max(kp1[x[0].queryIdx].pt[1] for x in good))

# 获取img2左上角坐标
x2 = int(min(kp2[x[0].trainIdx].pt[0] for x in good))
y2 = int(min(kp2[x[0].trainIdx].pt[1] for x in good))
x3 = int(max(kp2[x[0].trainIdx].pt[0] for x in good))
y3 = int(max(kp2[x[0].trainIdx].pt[1] for x in good))

# 画矩形框
cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
# cv2.imshow('img1', img1)
cv2.rectangle(img, (x2 + img1.shape[1], y2), (x3 + img1.shape[1], y3), (0, 0, 255), 2)
# cv2.imshow('img2', img2)

cv2.imshow('SIFT', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
