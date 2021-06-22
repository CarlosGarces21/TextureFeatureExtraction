import numpy as np
import cv2
#for lamda in np.arange(0, np.pi, np.pi /16):
#    print(lamda)

for theta in range(8):  # Define number of thetas
    theta = theta / 4. * np.pi
    print(theta)

# count number of pixels

img = cv2.imread('Results/TestResultsSpot/15.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
b = cv2.inRange(img,100,255)
a = cv2.countNonZero(b)

print(a)

if a > 20000:
    print("yes")
else:
    print("no")

cv2.imshow('image',b)
cv2.waitKey(0)
