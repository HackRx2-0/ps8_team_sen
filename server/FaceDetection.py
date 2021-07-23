import numpy as np
import cv2
import dlib
from imutils import face_utils
import imutils
import time
import reference_world as world

class Detection():
    def __init__(self,image):
        #self.image = cv2.imread('/home/pradyumn/scripts/hackathons/Hackrx/main/backend/Image Dataset/Sunglasses.jpg')
        #self.image = cv2.imread('/home/pradyumn/scripts/hackathons/Hackrx/main/backend/Image Dataset/clear.png')
        
        #self.image = cv2.imread('/home/pradyumn/scripts/hackathons/Hackrx/main/backend/real.png')
        self.image = image
        self.grayImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.face_cascade = cv2.CascadeClassifier('/home/pradyumn/scripts/hackathons/Hackrx/main/backend/haarcascade/haarcascade_frontalface_default.xml')
        
        self.fitmentScore = -100


    def face(self): 
        #grayImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(self.grayImage) 
        countFace = faces.shape[0]
    
        if len(faces) == 0:
            print("ERROR!! \n No Face Detected")
            print("1. Fitment Score : ",self.fitmentScore)
            return ["No Face Detected", self.fitmentScore]
            
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
            print("Passed Criteria 1 : Face Detected")
            print("1. Fitment Score : ",self.fitmentScore)
            self.blur(roi_gray,roi_color)
        else:
            print("ERROR!! \n More than one face Detected")
            print("1. Fitment Score : ",self.fitmentScore)
            return ["More than one Face Detected", self.fitmentScore]
                
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        

    def blur(self,grayImage,colorImage):
        laplace = cv2.Laplacian(grayImage, cv2.CV_64F).var()
        threshold = 55.0 
        if laplace < threshold:
            print("ERROR!! \n Image is Blurred")
            print("2. Fitment Score : ",self.fitmentScore)
            return ["ERROR!! Image is Blurred", self.fitmentScore]

        else :
            print("Passed Criteria 2 : No Blur Detected")
            print("2. Fitment Score : ",self.fitmentScore)
            self.brightness(grayImage,colorImage)

    def brightness(self,grayImage,colorImage):
        L, A, B = cv2.split(cv2.cvtColor(colorImage, cv2.COLOR_BGR2LAB))
        L = L/np.max(L)
        #print(np.mean(L))
        if(np.mean(L) < 0.4):
            print("Error!! Image is Dark")
            print("3.Fitment Score : ", self.fitmentScore)
            return ["Error!! Image is Dark", self.fitmentScore]
        elif(np.mean(L) > 0.8):
            print("Error!! Image is Too Bright")
            print("3.Fitment Score : ", self.fitmentScore)
            return ["Error!! Image is Too Bright", self.fitmentScore]
        else:
            self.fitmentScore = self.fitmentScore+15
            print("Passed Criteria 3 : Image is Normal")
            print("3. Fitment Score : ",self.fitmentScore)
            self.eye(grayImage,colorImage)

    def eye(self,grayImage,colorImage):
        eye_cascade = cv2.CascadeClassifier('/home/pradyumn/scripts/hackathons/Hackrx/main/backend/haarcascade/haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(grayImage)
        #print(grayImage)
        #,scaleFactor=1.04,minNeighbors=13,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        if len(eyes)==0:
            print("Error!! No eyes detected")
            print("4.Fitment Score : ",self.fitmentScore)
            return ["Error!! No eyes detected", self.fitmentScore]
        elif eyes.shape[0]==2:
            self.fitmentScore=self.fitmentScore+20
            print("Passed Criteria 4 : Detected Eyes")
            print("4. Fitment Score : ",self.fitmentScore)
            self.mouth(grayImage,colorImage)
        else:
            print("Error!! Eyes are covered")
            print("4. Fitment Score : ",self.fitmentScore)
            return ["Error!! Eyes are covered",self.fitmentScore]


    def mouth(self, grayImage,colorImage): 
        mouth_cascade = cv2.CascadeClassifier('/home/pradyumn/scripts/hackathons/Hackrx/main/backend/haarcascade/haarcascade_smile.xml')
        mouth = mouth_cascade.detectMultiScale(grayImage)
                #,scaleFactor=1.4,minNeighbors=26,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE) 
        countMouth = mouth.shape[0]
        if countMouth == 1:
            self.fitmentScore = self.fitmentScore+20
            print("Passed Criteria 5 : Face is not covered")
            print("5. Fitment Score : ",self.fitmentScore)
            self.pose(grayImage,colorImage)
        else :
            print("Error!! Face is covered")
            print("5. Fitment Score : ",self.fitmentScore)
            return ["Error!! Face is covered",self.fitmentScore]


    def pose(self,grayImage,colorImage):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
        #im = cv2.imread('/home/pradyumn/scripts/hackathons/Hackrx/main/backend/Image Dataset/clear.png')
        faces = detector(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB), 0)
        face3Dmodel = world.ref3DModel()
        for face in faces:
            shape = predictor(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB), face)
            refImgPts = world.ref2dImagePoints(shape)
            height, width, channel = self.image.shape
            focalLength = 1 * width
            cameraMatrix = world.cameraMatrix(focalLength, (height / 2, width / 2))
            mdists = np.zeros((4, 1), dtype=np.float64)
            # calculate rotation and translation vector using solvePnP
            success, rotationVector, translationVector = cv2.solvePnP(
                face3Dmodel, refImgPts, cameraMatrix, mdists)
            noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
            noseEndPoint2D, jacobian = cv2.projectPoints(
                noseEndPoints3D, rotationVector, translationVector, cameraMatrix, mdists)
            # draw nose line 
            p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
            p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))
            # calculating angle
            rmat, jac = cv2.Rodrigues(rotationVector)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            x = np.arctan2(Qx[2][1], Qx[2][2])
            y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1] ) + (Qy[2][2] * Qy[2][2])))
            z = np.arctan2(Qz[0][0], Qz[1][0])
            gaze = "Looking: "
            if angles[1] < -15:
                self.fitmentScore = self.fitmentScore-10
                print("Looking Left")
                print("6. Fitment Score : ", self.fitmentScore)
                return ["Looking Left",self.fitmentScore]
            elif angles[1] > 15:
                self.fitmentScore = self.fitmentScore-10
                print("Looking Right")
                print("6. Fitment Score : ", self.fitmentScore)
                return ["Looking Right",self.fitmentScore]
            else:
                self.fitmentScore = self.fitmentScore+20
                print("Looking Forward")
                print("6. Fitment Score : ", self.fitmentScore)
                return ["Normal Image",self.fitmentScore]
if __name__ == "__main__":
    image = cv2.imread('/home/pradyumn/scripts/hackathons/Hackrx/main/backend/Image Dataset/clear.png')
    #obj = Detection(image)
    #obj.face()
