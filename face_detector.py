# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:00:19 2019

@author: Sadanand Vishwas
"""

# impoorting the libraries
import cv2
import numpy as np
import sys

# import the  front face detection Haar-cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# import the  eye detection Haar-cascade classifier
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

# Now the fun part begins
# lets check for the video feed input
# if no arguments is passed then take feed from the default webcam
if len(sys.argv) == 1:
    # default video capture device
    capture_device = 0
else:
    # user prompted video capture device
    capture_device = sys.argv[1]

# Initialize the selected video capture device
video_feed = cv2.VideoCapture(capture_device)

# function to detect faces and eyes on a picture
def faceDetect(img_frame):
    # Lets convert the RGB image to grayscale for easier computation as it has only 
    # two pixel values, black, and white
    gray_scale_img = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
    
    # Now let's process the gray_scaled image on face detection cascade to get the
    #  face boundries on the image
    faces = face_cascade.detectMultiScale(gray_scale_img, scaleFactor= 1.3,
                                          minNeighbors=5)
    
    # now let's process each face in the faces to detect the eyes in the face
    # (x, y) is the starting coordinate of the face rectangle and
    # w and h are width and height of the face rectangle
    for face in faces:
        x, y, w, h = face
        # Now we will create a rectangle around the face img_frame
        # i.e our ROI(region of interest) to mark the face as detected
        cv2.rectangle(img_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # now lets get the roi portion of both gray and colored images to detect eyes
        roi_gray = gray_scale_img[y: y + h, x: x + w]
        roi_color = img_frame[y: y + h, x: x + w]
        
        # now the eye always lies on the face so we try to detect the eyes with
        # Haar eye-cascde classifier
        eyes = eye_cascade.detectMultiScale(roi_gray)
        # now lets process each eye on the face
        # (ex, ey) is the starting coordinate of the eye rectangle and
        # ew and eh are width and height of the eye rectangle
        for eye in eyes:
            ex, ey, ew, eh = eye
            # Now draw the rectange around our ROI
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 240, 0), 2)
            # Voila, you have successfully done it
    
    # Now lets show the detected faces and eyes
    cv2.imshow('Image Output', img_frame)

# Infinite loop for detection of face and eyes on realtime video feed
while True:
    # Read the video feed to get a image frame from it to process
    retval, img_frame = video_feed.read()
    if not retval:
        print("Unable to capture frame from video feed")
        cv2.destroyAllWindows()
        exit(1)
    
    # detect the face and eyes from the image frame
    faceDetect(img_frame)
    
    # press `q` to exit the program
    key = cv2.waitKey(1)
    if key == ord('q'):
            # release the video capturing object
            video_feed.release()
            # destroy all the windows opened
            cv2.destroyAllWindows()
            exit(0)

print("Model executed successfully")
    
        