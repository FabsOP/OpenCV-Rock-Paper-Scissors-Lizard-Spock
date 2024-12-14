import cv2
import cvzone
from cvzone import HandTrackingModule

################## SET UP WEBCAM ######################
cam = cv2.VideoCapture(0)

cam.set(3,960)                        #camera width
cam.set(4,540)                        #camera height

while True:
    ############## Camera ##################################
    #get camera img
    success, img = cam.read()       #return success boolean and image
    
    #resize camera img 205/540
    img = cv2.resize(img, (0,0), None, 0.379, 0.379)
    
    #Flip camera accross the y-axis to mirror user correctly
    img = cv2.flip(img, 1)
    
    #Crop camera img to fix in player box
    img = img[:,80:283]
        
    #get background img and resize
    imgBG = cv2.imread("UI/bg.png")
    imgBG = cv2.resize(imgBG, (960, 540)) 
    
    
    ########## Place Camera img on bg img ###############
    imgBG[199:404,598:801] = img
    
    ########## Show images ####################
    #Show camera img
    cv2.imshow("You", img)
    
    #show background img
    cv2.imshow("bg", imgBG)
    
    ################# End of Loop ###########
    #refresh frequency
    cv2.waitKey(1)