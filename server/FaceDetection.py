import numpy as np
import cv2
#import dlib
#import imutils
import time

class Detection():
    '''
    def __init__(self):
        self.image = cv2.imread('C:/Users/raahi/Desktop/HackerRx/face1.jpeg')
        #self.image = image
        self.face_cascade = cv2.CascadeClassifier('C:/Users/raahi/Desktop/HackerRx\haarcascade_frontalface_default.xml')
'''
    def face():
        image = cv2.imread('C:/Users/raahi/Desktop/HackerRx/face1.jpeg')
        #self.image = image
        face_cascade = cv2.CascadeClassifier('C:/Users/raahi/Desktop/HackerRx\haarcascade_frontalface_default.xml')
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(grayImage)

        print(type(faces))
        countFace = faces.shape[0]
        
        if len(faces) == 0:
            print("No faces found")

        elif(countFace==1):
            print("One Face Detected!! \n Selecting ROI")
            for (x,y,w,h) in faces:
                x=x-150
                y=y-200
                w=w+300
                h=h+300
                roi_gray = grayImage[y:y+h,x:x+w]
                roi_color = image[y:y+h,x:x+w]
            cv2.imshow('ROI',roi_color)
        else:
            print("More than one face detected")
                
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    obj = Detection()
    obj.face()
