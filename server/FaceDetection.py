import numpy as np
import cv2
import dlib
from imutils import face_utils
import imutils
import time

class Detection():
    def __init__(self):
        self.image = cv2.imread('/home/pradyumn/scripts/hackathons/Hackrx/images/face1.png')
        #self.image = image
        self.grayImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.face_cascade = cv2.CascadeClassifier('/home/pradyumn/scripts/hackathons/Hackrx/main/backend/haarcascade/haarcascade_frontalface_default.xml')
        
        self.fitmentScore = 0


    def face(self):
        #grayImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(self.grayImage) 
        countFace = faces.shape[0]
    
        if len(faces) == 0:
            print("No faces found")
            self.fitmentScore = self.fitmentScore-10
            return ["No Face Found", self.fitmentScore]
            
        elif(countFace==1):
            print("One Face Detected!! \n Selecting ROI")
            for (x,y,w,h) in faces:
                x=x-150
                y=y-200
                w=w+300
                h=h+300
                roi_gray = self.grayImage[y:y+h,x:x+w]
                roi_color = self.image[y:y+h,x:x+w]
            #cv2.imshow('ROI',roi_color)
            self.fitmentScore = self.fitmentScore+20

            #self.blur(roi_gray)
            #self.brightness(roi_color)
            #self.mouth(roi_gray)
            #self.eyes(roi_gray)
            #self.landmarks(roi_gray)
        else:
            print("More than one face detected")
            self.fitmentScore = self.fitmentScore-20
            return ["More than one Face Detected", self.fitmentScore]
                
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def blur(self,grayImage):
        laplace = cv2.Laplacian(grayImage, cv2.CV_64F).var()
        threshold = 100.0
        text = "Not Blurry"
        
        # if the focus measure is less than the supplied threshold,
        # then the image should be considered "blurry"
        if laplace < threshold:
            self.fitmentScore = self.fitmentScore-20
            return ["Blurred Image!!", self.fitmentScore]

        else :
            self.fitmentScore=self.fitmentScore +20
            #self.brightness()

    def brightness(self,colorImage):
        L, A, B = cv2.split(cv2.cvtColor(colorImage, cv2.COLOR_BGR2LAB))

        # Normalize L channel by dividing all pixel values with maximum pixel value
        L = L/np.max(L)
        print(np.mean(L))
        # Return True if mean is greater than thresh else False
        if(np.mean(L) < 0.5):
            print("Dark")
            self.fitmentScore = self.fitmentScore-20
            return ["Image is Dark", self.fitmentScore]
        elif(np.mean(L) > 1.0):
            self.fitmentScore = self.fitmentScore-20
            print('Too bright')
            return ["Image is too dark",self.fitmetnScore]
        else:
            print('Normal')
            self.fitmentScore=self.fitmentScore+20


        #labim = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
        #cv2.imwrite('./imgLAB.jpg',labim)
    
    def eyes(self,grayImage):
        eye_cascade = cv2.CascadeClassifier('/home/pradyumn/scripts/hackathons/Hackrx/haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(grayImage,scaleFactor=1.04,minNeighbors=13,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        if len(eyes)==0:
            print("No eyes detected")
        for (x,y,w,h) in eyes:
            cv2.rectangle(grayImage,(x,y),(x+w,y+h),(0,255,255),1)
        print(eyes.shape[0])
        cv2.imshow('Eyes',grayImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



    def mouth(self, grayImage): 
        #image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        mouth_cascade = cv2.CascadeClassifier('/home/pradyumn/scripts/hackathons/Hackrx/haarcascade_smile.xml')
        mouth = mouth_cascade.detectMultiScale(grayImage,scaleFactor=1.4,minNeighbors=26,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE) 

        #for (x,y,w,h) in mouth:
        #    y = int(y - 0.17*h)
        #    cv2.rectangle(grayImage,(x,y),(x+w,y+h),(0,0,0),1)
        #cv2.imshow("Mouth",grayImage)
        #print(mouth.shape[0])
        countMouth = mouth.shape[0]
        if countMouth == 1:
            self.fitmentScore = self.fitmentScore+20
        else :
            self.fitmentScore = self.fitmentScore-20
            return ["Face is covered",self.fitmentScore]


    def landmarks(self,grayImage):
        path = "./shape_predictor_68_face_landmarks.dat"
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(path)

        # load the input image, resize it, and convert it to grayscale
        #image = imutils.resize(image, width=500)
        print("Inside")
        rects = detector(grayImage, 1)

        # loop over the face detections
        for (i, rect) in enumerate(rects):
            shape = predictor(grayImage, rect)
            shape = face_utils.shape_to_np(shape)

            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # show the face number
            cv2.putText(grayImage, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                    cv2.circle(grayImage, (x, y), 1, (0, 0, 255), -1)
        
        cv2.imshow("Output", grayImage)

if __name__ == "__main__":
    obj = Detection()
    obj.face()
