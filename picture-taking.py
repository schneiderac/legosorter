import numpy as np  # importing Numpy for use w/ OpenCV
import cv2  # importing Python OpenCV
from datetime import datetime  # importing datetime for naming files w/ timestamp
import time

offset = 230

def diffImg(t0, t1, t2):  # Function to calculate difference between images.
    d1 = cv2.absdiff(t2, t1)
    d2 = cv2.absdiff(t1, t0)
    return cv2.bitwise_and(d1, d2)


threshold = 250  # Threshold for triggering "motion detection"
cam = cv2.VideoCapture(0)  # Lets initialize capture on webcam

winName = "Movement Indicator"  # comment to hide window
cv2.namedWindow(winName)  # comment to hide window

# Read three images first:
t_minus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
t = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)

#
# t_minus = cv2.blur(t_minus,(100,100))
# t = cv2.blur(t,(100,100))
# t_plus = cv2.blur(t_plus,(100,100))

# t_minus = cv2.adaptiveThreshold(t_minus, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)
# t = cv2.adaptiveThreshold(t, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)
# t_plus = cv2.adaptiveThreshold(t_plus, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)

# Lets use a time check so we only take 1 pic per sec
timeCheck = datetime.now().strftime('%Ss')

while True:
    ret, frame = cam.read()  # read from camera
    diff = diffImg(t_minus, t, t_plus)
    totalDiff = cv2.countNonZero(diff[0:480, 400:640])  # this is total difference number
    text = "threshold: " + str(totalDiff)  # make a text showing total diff.
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # display it on screen
    if totalDiff > threshold: # and timeCheck != datetime.now().strftime('%Ss'):
        time.sleep(0.05)
        dimg = cam.read()[1]
        cv2.imwrite('/home/andreas/motion/' + datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f') + '.jpg', dimg)
        time.sleep(0.1)
        t_minus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
        t = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
        t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
        t_minus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
        t = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
        t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)

    timeCheck = datetime.now().strftime('%Ss')

    # Read next image
    t_minus = t
    t = t_plus
    t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)

    t_minus = cv2.blur(t_minus,(100,100))
    t = cv2.blur(t,(100,100))
    t_plus = cv2.blur(t_plus,(100,100))

    t_minus = cv2.adaptiveThreshold(t_minus, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)
    t = cv2.adaptiveThreshold(t, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)
    t_plus = cv2.adaptiveThreshold(t_plus, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)

    cv2.imshow(winName, frame)

    cv2.imshow('hello', t_minus)

    key = cv2.waitKey(10)
    if key == 27:  # comment this 'if' to hide window
        cv2.destroyWindow(winName)
        break
