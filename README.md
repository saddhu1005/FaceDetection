# Face & Eye Detection using OpenCV Haar-Cascades
Any object detection technique uses classifiers, here we are going to use the Opencv provided pre-trained Haar-cascades to detect face and eye of a human being. We can also use the LBP-cascades which is also provided by the OpenCV library to detect any object, or we can train and build our own cascade classifiers.
Cascade classifiers are integration of multiple classifiers, thus the name cascades. Classifiers are alone weak and gives less accuracy on detection but when classifiers of different features combined together, it gives higher accuracy and stablity.
The training of the cascade-classifiers are done by taking 2 types of data, one is the positive data which contains the images of the objects we need to detect, and the other is the negetive data which does not contain the relative objects we need to detect.
By taking this two class of data we train our classifiers to create vector of features and merge them to create the cascades ('xml' files).

This model can both work on a video source feed or real time video feed from webcam of a computer. We take a frame at a time from the video feed and process it with the Haar-cascade classifiers to detect the boundaries of face and eyes of a human being.

## Test Run
![test_image](https://github.com/saddhu1005/FaceDetection/blob/master/outputs/test_image.png)
