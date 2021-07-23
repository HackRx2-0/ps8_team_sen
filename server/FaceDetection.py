import numpy as np
import cv2
import dlib
from imutils import face_utils
import imutils
import time

class Detection():
    def __init__(self):
        self.image = cv2.imread('/home/pradyumn/scripts/hackathons/Hackrx/main/backend/Image Dataset/Sunglasses.jpg')
        #self.image = cv2.imread('/home/pradyumn/scripts/hackathons/Hackrx/main/backend/test pics/Blurry.jpg')
        
        #self.image = cv2.imread('/home/pradyumn/scripts/hackathons/Hackrx/main/backend/real.png')
        #self.image = image
        self.grayImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.face_cascade = cv2.CascadeClassifier('/home/pradyumn/scripts/hackathons/Hackrx/main/backend/haarcascade/haarcascade_frontalface_default.xml')
        
        self.fitmentScore = -100


    def face(self):
        
        #grayImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(self.grayImage) 
        countFace = faces.shape[0]
    
        if len(faces) == 0:
            print("No faces found")
            #self.fitmentScore = self.fitmentScore-20
            return ["No Face Found", self.fitmentScore]
            
        elif(countFace==1):
            #print("One Face Detected!! \n Selecting ROI")
            for (x,y,w,h) in faces:
                #x=x-150
                #y=y-200
                #w=w+300
                #h=h+300
                roi_gray = self.grayImage[y:y+h,x:x+w]
                roi_color = self.image[y:y+h,x:x+w]
            cv2.imwrite("./roi.png",roi_color)
            self.fitmentScore = self.fitmentScore+20
            print("Face Detected", self.fitmentScore)
            self.blur(roi_gray,roi_color)
            #self.eye(roi_gray,roi_color)
        else:
            print("More than one face detected")
            #self.fitmentScore = self.fitmentScore-20
            print("More than one face Detected",self.fitmentScore)
            return ["More than one Face Detected", self.fitmentScore]
                
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    
    def blur(self,grayImage,colorImage):
        laplace = cv2.Laplacian(grayImage, cv2.CV_64F).var()
        threshold = 55.0 
        if laplace < threshold:
            #self.fitmentScore = self.fitmentScore-20
            print("Blurred Image",self.fitmentScore)
            print(laplace)
            return ["Blurred Image!!", self.fitmentScore]

        else :
            self.fitmentScore=self.fitmentScore +15
            print("Passed Blur Criteria", self.fitmentScore)
            self.brightness(grayImage,colorImage)

    def brightness(self,grayImage,colorImage):
        L, A, B = cv2.split(cv2.cvtColor(colorImage, cv2.COLOR_BGR2LAB))
        L = L/np.max(L)
        #print(np.mean(L))
        if(np.mean(L) < 0.4):
            #self.fitmentScore = self.fitmentScore-20
            print("Image is Dark", self.fitmentScore)
            return ["Image is Dark", self.fitmentScore]
        elif(np.mean(L) > 0.8):
            #self.fitmentScore = self.fitmentScore-20
            print('Image is too Bright', self.fitmentScore)
            return ["Image is too Bright",self.fitmetnScore]
        else:
            self.fitmentScore = self.fitmentScore+15
            print('Image is Normal',self.fitmentScore)
            self.eye(grayImage,colorImage)

    def eye(self,grayImage,colorImage):
        eye_cascade = cv2.CascadeClassifier('/home/pradyumn/scripts/hackathons/Hackrx/main/backend/haarcascade/haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(grayImage)
        #print(grayImage)
        #,scaleFactor=1.04,minNeighbors=13,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        if len(eyes)==0:
            #self.fitmentScore=self.fitmentScore-20
            print("No eyes detected",self.fitmentScore)
        elif eyes.shape[0]==2:
            self.fitmentScore=self.fitmentScore+20
            print("Eyes Detected",self.fitmentScore)
            self.mouth(grayImage,colorImage)
        else:
            #self.fitmentScore=self.fitmentScore-20
            print("Eyes are covered",self.fitmentScore)


    def mouth(self, grayImage,colorImage): 
        mouth_cascade = cv2.CascadeClassifier('/home/pradyumn/scripts/hackathons/Hackrx/main/backend/haarcascade/haarcascade_smile.xml')
        mouth = mouth_cascade.detectMultiScale(grayImage)
                #,scaleFactor=1.4,minNeighbors=26,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE) 
        countMouth = mouth.shape[0]
        if countMouth == 1:
            self.fitmentScore = self.fitmentScore+20
            print("Face is not Covered", self.fitmentScore)
            self.landmarks(grayImage,colorImage)
        else :
            #self.fitmentScore = self.fitmentScore-20
            print("Face is covered",self.fitmentScore)
            return ["Face is covered",self.fitmentScore]


    def landmarks(self,grayImage,colorImage):
        path = "./shape_predictor_68_face_landmarks.dat"
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(path)
        rects = detector(grayImage, 1)
        for (i, rect) in enumerate(rects):
            shape = predictor(grayImage, rect)
            shape = face_utils.shape_to_np(shape)
            (x, y, w, h) = face_utils.rect_to_bb(rect)
        self.fitmentScore=self.fitmentScore+20
        print("Landmarks Recieved",self.fitmentScore)

if __name__ == "__main__":
    obj = Detection()
    obj.face()
